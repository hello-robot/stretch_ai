# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import os
import pickle
import threading

# import wget
import time
from io import BytesIO
from pathlib import Path

import clip
import cv2
import numpy as np
import rerun as rr
import scipy
import torch

# from stretch.utils.morphology import get_edges
import torch.nn.functional as F
import wget
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

# from segment_anything import sam_model_registry, SamPredictor
# from transformers import AutoProcessor, OwlViTForObjectDetection
from ultralytics import YOLOWorld

from stretch.core import get_parameters
from stretch.dynav.communication_util import load_socket, recv_everything
from stretch.dynav.mapping_utils.a_star import AStar
from stretch.dynav.mapping_utils.voxel import SparseVoxelMap
from stretch.dynav.mapping_utils.voxel_map import SparseVoxelMapNavigationSpace
from stretch.dynav.scannet import CLASS_LABELS_200
from stretch.dynav.voxel_map_localizer import VoxelMapLocalizer


def get_inv_intrinsics(intrinsics):
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = (
        intrinsics[..., 0, 0],
        intrinsics[..., 1, 1],
        intrinsics[..., 0, 2],
        intrinsics[..., 1, 2],
    )
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


def get_xyz(depth, pose, intrinsics):
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """
    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth)
    if not isinstance(pose, torch.Tensor):
        pose = torch.from_numpy(pose)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.from_numpy(intrinsics)
    while depth.ndim < 4:
        depth = depth.unsqueeze(0)
    while pose.ndim < 3:
        pose = pose.unsqueeze(0)
    while intrinsics.ndim < 3:
        intrinsics = intrinsics.unsqueeze(0)
    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    xyz = xyz.unflatten(1, (height, width))

    return xyz


class ImageProcessor:
    def __init__(
        self,
        vision_method="mask*lip",
        siglip=True,
        device="cuda",
        min_depth=0.25,
        max_depth=2.5,
        img_port=5558,
        text_port=5556,
        open_communication=True,
        rerun=True,
        static=True,
        log=None,
        image_shape=(450, 350),
    ):
        self.static = static
        self.siglip = siglip
        current_datetime = datetime.datetime.now()
        if log is None:
            self.log = "debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = log
        self.rerun = rerun
        if self.rerun:
            if self.static:
                rr.init(self.log, spawn=False)
                rr.connect("100.108.67.79:9876")
            else:
                rr.init(self.log, spawn=True)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.obs_count = 0
        # There are three vision methods:
        # 1. 'mask*lip' Following the idea of https://arxiv.org/abs/2112.01071, remove the last layer of any VLM and obtain the dense features
        # 2. 'mask&*lip' Following the idea of https://mahis.life/clip-fields/, extract segmentation mask and assign a vision-language feature to it
        self.vision_method = vision_method
        # If cuda is not available, then device will be forced to be cpu
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.create_obstacle_map()
        self.create_vision_model()

        self.voxel_map_lock = (
            threading.Lock()
        )  # Create a lock for synchronizing access to `self.voxel_map_localizer`

        self.traj = None
        self.image_shape = image_shape

        if open_communication:
            self.img_socket = load_socket(img_port)
            self.text_socket = load_socket(text_port)

            self.img_thread = threading.Thread(target=self._recv_image)
            self.img_thread.daemon = True
            self.img_thread.start()

    def create_obstacle_map(self):
        print("- Load parameters")
        parameters = get_parameters("dynav_config.yaml")
        self.default_expand_frontier_size = parameters["default_expand_frontier_size"]
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            exp_min_density=parameters["exp_min_density"],
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            # step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )

        # Create a simple motion planner
        self.planner = AStar(self.space)
        self.value_map = torch.zeros(self.voxel_map.grid_size)

    def create_vision_model(self):
        if not self.siglip:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
                self.device
            )
            self.clip_preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.clip_model.eval()
        if self.vision_method != "mask*lip":
            sam_checkpoint = f"./sam2_hiera_small.pt"
            sam_config = "sam2_hiera_s.yaml"
            if not os.path.exists(sam_checkpoint):
                wget.download(
                    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
                    out=sam_checkpoint,
                )
            sam2_model = build_sam2(
                sam_config, sam_checkpoint, device=self.device, apply_postprocessing=False
            )
            self.mask_predictor = SAM2ImagePredictor(sam2_model)
        self.yolo_model = YOLOWorld("yolov8s-worldv2.pt")
        self.texts = CLASS_LABELS_200
        self.yolo_model.set_classes(self.texts)
        self.voxel_map_localizer = VoxelMapLocalizer(
            self.voxel_map,
            clip_model=self.clip_model,
            processor=self.clip_preprocess,
            device=self.device,
            siglip=self.siglip,
        )

    def process_text(self, text, start_pose):
        """
        Process the text query and return the trajectory for the robot to follow.
        """

        if self.rerun:
            rr.log("/object", rr.Clear(recursive=True), static=self.static)
            rr.log("/robot_start_pose", rr.Clear(recursive=True), static=self.static)
            rr.log("/direction", rr.Clear(recursive=True), static=self.static)
            rr.log("robot_monologue", rr.Clear(recursive=True), static=self.static)
            rr.log(
                "/Past_observation_most_similar_to_text",
                rr.Clear(recursive=True),
                static=self.static,
            )
            if not self.static:
                rr.connect("100.108.67.79:9876")
        debug_text = ""
        mode = "navigation"
        obs = None
        localized_point = None
        waypoints = None

        if text is not None and text != "" and self.traj is not None:
            print("saved traj", self.traj)
            traj_target_point = self.traj[-1]
            if self.voxel_map_localizer.verify_point(text, traj_target_point):
                localized_point = traj_target_point
                debug_text += "## Last visual grounding results looks fine so directly use it.\n"

        if waypoints is None:
            # Do visual grounding
            if text is not None and text != "" and localized_point is None:
                with self.voxel_map_lock:
                    (
                        localized_point,
                        debug_text,
                        obs,
                        pointcloud,
                    ) = self.voxel_map_localizer.localize_A(text, debug=True, return_debug=True)
                if localized_point is not None:
                    rr.log(
                        "/object",
                        rr.Points3D(
                            [localized_point[0], localized_point[1], 1.5],
                            colors=torch.Tensor([1, 0, 0]),
                            radii=0.1,
                        ),
                        static=self.static,
                    )
            # Do Frontier based exploration
            if text is None or text == "" or localized_point is None:
                debug_text += "## Navigation fails, so robot starts exploring environments.\n"
                localized_point = self.sample_frontier(start_pose, text)
                mode = "exploration"
                rr.log(
                    "/object",
                    rr.Points3D([0, 0, 0], colors=torch.Tensor([1, 0, 0]), radii=0),
                    static=self.static,
                )
                print("\n", localized_point, "\n")

            if localized_point is None:
                return []

            if len(localized_point) == 2:
                localized_point = np.array([localized_point[0], localized_point[1], 0])

            point = self.sample_navigation(start_pose, localized_point)

            if self.rerun:
                buf = BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)
                img = np.array(img)
                buf.close()
                rr.log("2d_map", rr.Image(img), static=self.static)
            else:
                if text != "":
                    plt.savefig(self.log + "/debug_" + text + str(self.obs_count) + ".png")
                else:
                    plt.savefig(self.log + "/debug_exploration" + str(self.obs_count) + ".png")
            plt.clf()

            if obs is not None and mode == "navigation":
                rgb = self.voxel_map.observations[obs - 1].rgb
                if not self.rerun:
                    if isinstance(rgb, torch.Tensor):
                        rgb = np.array(rgb)
                    cv2.imwrite(self.log + "/debug_" + text + ".png", rgb[:, :, [2, 1, 0]])
                else:
                    rr.log(
                        "/Past_observation_most_similar_to_text", rr.Image(rgb), static=self.static
                    )
            waypoints = None

            if point is None:
                print("Unable to find any target point, some exception might happen")
            else:
                print("Target point is", point)
                res = self.planner.plan(start_pose, point)
            if res.success:
                waypoints = [pt.state for pt in res.trajectory]
            else:
                waypoints = None
                print("[FAILURE]", res.reason)
        # If we are navigating to some object of interest, send (x, y, z) of
        # the object so that we can make sure the robot looks at the object after navigation
        traj = []
        if waypoints is not None:
            finished = len(waypoints) <= 10 and mode == "navigation"
            if finished:
                self.traj = None
            else:
                self.traj = waypoints[8:] + [[np.nan, np.nan, np.nan], localized_point]
            if not finished:
                waypoints = waypoints[:8]
            traj = self.planner.clean_path_for_xy(waypoints)
            if finished:
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            print("Planned trajectory:", traj)

        if self.rerun:
            if text is not None and text != "":
                debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
            else:
                debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
            debug_text = "# Robot's monologue: \n" + debug_text
            rr.log(
                "robot_monologue",
                rr.TextDocument(debug_text, media_type=rr.MediaType.MARKDOWN),
                static=self.static,
            )

        if traj is not None:
            origins = []
            vectors = []
            for idx in range(len(traj)):
                if idx != len(traj) - 1:
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append(
                        [traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0]
                    )
            rr.log(
                "/direction",
                rr.Arrows3D(
                    origins=origins, vectors=vectors, colors=torch.Tensor([0, 1, 0]), radii=0.1
                ),
                static=self.static,
            )
            rr.log(
                "/robot_start_pose",
                rr.Points3D(
                    [start_pose[0], start_pose[1], 1.5], colors=torch.Tensor([0, 0, 1]), radii=0.1
                ),
                static=self.static,
            )

        # self.write_to_pickle()
        return traj

    def sample_navigation(self, start, point, mode="navigation"):
        # plt.clf()
        obstacles, _ = self.voxel_map.get_2d_map()
        plt.imshow(obstacles)
        if point is None:
            start_pt = self.planner.to_pt(start)
            # plt.scatter(start_pt[1], start_pt[0], s = 10)
            return None
        goal = self.space.sample_target_point(
            start, point, self.planner, exploration=mode != "navigation"
        )
        print("point:", point, "goal:", goal)
        obstacles, explored = self.voxel_map.get_2d_map()
        start_pt = self.planner.to_pt(start)
        plt.scatter(start_pt[1], start_pt[0], s=15, c="b")
        point_pt = self.planner.to_pt(point)
        plt.scatter(point_pt[1], point_pt[0], s=15, c="r")
        if goal is not None:
            goal_pt = self.planner.to_pt(goal)
            plt.scatter(goal_pt[1], goal_pt[0], s=10, c="g")
        return goal

    def sample_frontier(self, start_pose=[0, 0, 0], text=None):
        with self.voxel_map_lock:
            if text is not None and text != "":
                (
                    index,
                    time_heuristics,
                    alignments_heuristics,
                    total_heuristics,
                ) = self.space.sample_exploration(
                    start_pose, self.planner, self.voxel_map_localizer, text, debug=False
                )
            else:
                index, time_heuristics, _, total_heuristics = self.space.sample_exploration(
                    start_pose, self.planner, None, None, debug=False
                )
                alignments_heuristics = time_heuristics

        obstacles, explored = self.voxel_map.get_2d_map()
        # plt.clf()
        # plt.imshow(obstacles * 0.5 + explored * 0.5)
        # plt.scatter(index[1], index[0], s = 20, c = 'r')
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))

    def _recv_image(self):
        while True:
            # data = recv_array(self.img_socket)
            rgb, depth, intrinsics, pose = recv_everything(self.img_socket)
            print("Image received")
            start_time = time.time()
            self.process_rgbd_images(rgb, depth, intrinsics, pose)
            process_time = time.time() - start_time
            print("Image processing takes", process_time, "seconds")
            print("processing took " + str(process_time) + " seconds")

    def forward_one_block(self, resblocks, x):
        q, k, v = None, None, None
        y = resblocks.ln_1(x)
        y = F.linear(y, resblocks.attn.in_proj_weight, resblocks.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, resblocks.attn.out_proj.weight, resblocks.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + resblocks.mlp(resblocks.ln_2(v))

        return v

    def forward_one_block_siglip(self, resblocks, x):
        q, k, v = None, None, None
        x = F.linear(x, resblocks.in_proj_weight, resblocks.in_proj_bias)
        N, L, C = x.shape
        x = x.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        x = F.linear(x, resblocks.out_proj.weight, resblocks.out_proj.bias)
        q, k, v = x.tensor_split(3, dim=0)

        return v

    def extract_mask_clip_features(self, x, image_shape):
        if self.siglip:
            with torch.no_grad():
                output = self.clip_model.vision_model(x["pixel_values"], output_hidden_states=True)
            feat = output.last_hidden_state
            feat = self.forward_one_block_siglip(self.clip_model.vision_model.head.attention, feat)
            feat = self.clip_model.vision_model.head.layernorm(feat)
            feat = feat + self.clip_model.vision_model.head.mlp(feat)
            feat = feat.detach().cpu()
            with torch.no_grad():
                N, L, H, W = self.clip_model.vision_model.embeddings.patch_embedding(
                    x["pixel_values"]
                ).shape
            feat = feat.reshape(N, H, W, L).permute(0, 3, 1, 2)
        else:
            with torch.no_grad():
                x = self.clip_model.visual.conv1(x)
                N, L, H, W = x.shape
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                x = torch.cat(
                    [
                        self.clip_model.visual.class_embedding.to(x.dtype)
                        + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                        x,
                    ],
                    dim=1,
                )
                x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
                x = self.clip_model.visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                for idx in range(self.clip_model.visual.transformer.layers):
                    if idx == self.clip_model.visual.transformer.layers - 1:
                        break
                    x = self.clip_model.visual.transformer.resblocks[idx](x)
                x = self.forward_one_block(self.clip_model.visual.transformer.resblocks[-1], x)
                x = x[1:]
                x = x.permute(1, 0, 2)
                x = self.clip_model.visual.ln_post(x)
                x = x @ self.clip_model.visual.proj
                feat = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode="bilinear", align_corners=True)
        feat = F.normalize(feat, dim=1)
        return feat.permute(0, 2, 3, 1)

    def run_mask_clip(self, rgb, mask, world_xyz):

        with torch.no_grad():
            results = self.yolo_model.predict(
                rgb.permute(1, 2, 0)[:, :, [2, 1, 0]].numpy(), conf=0.1, verbose=False
            )
            xyxy_tensor = results[0].boxes.xyxy
            if len(xyxy_tensor) == 0:
                return
            bounding_boxes = torch.stack(
                sorted(
                    xyxy_tensor, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
                ),
                dim=0,
            )
            bbox_mask = torch.zeros_like(mask)
            for box in bounding_boxes:
                tl_x, tl_y, br_x, br_y = box
                bbox_mask[
                    max(int(tl_y), 0) : min(int(br_y), rgb.shape[1]),
                    max(int(tl_x), 0) : min(int(br_x), rgb.shape[2]),
                ] = 1
            bbox_mask = bbox_mask.bool()
            # print(mask, bbox_mask)
            mask = torch.logical_or(mask, ~bbox_mask)

            if not self.siglip:
                if self.device == "cpu":
                    input = (
                        self.clip_preprocess(transforms.ToPILImage()(rgb))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                else:
                    input = (
                        self.clip_preprocess(transforms.ToPILImage()(rgb))
                        .unsqueeze(0)
                        .to(self.device)
                        .half()
                    )
            else:
                input = self.clip_preprocess(images=rgb, padding="max_length", return_tensors="pt")
                for i in input:
                    input[i] = input[i].to(self.device)
            features = self.extract_mask_clip_features(input, rgb.shape[-2:])[0].cpu()

        valid_xyz = world_xyz[~mask]
        features = features[~mask]
        valid_rgb = rgb.permute(1, 2, 0)[~mask]
        if len(valid_xyz) != 0:
            self.add_to_voxel_pcd(valid_xyz, features, valid_rgb)

    def run_owl_sam_clip(self, rgb, mask, world_xyz):
        with torch.no_grad():
            results = self.yolo_model.predict(
                rgb.permute(1, 2, 0)[:, :, [2, 1, 0]].numpy(), conf=0.1, verbose=False
            )
            xyxy_tensor = results[0].boxes.xyxy
            if len(xyxy_tensor) == 0:
                return

            self.mask_predictor.set_image(rgb.permute(1, 2, 0).numpy())
            # bounding_boxes = torch.stack(sorted(results[0]['boxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
            bounding_boxes = torch.stack(
                sorted(
                    xyxy_tensor, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
                ),
                dim=0,
            )
            # transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(bounding_boxes.detach().to(self.device), rgb.shape[-2:])
            # masks, _, _= self.mask_predictor.predict_torch(
            masks, _, _ = self.mask_predictor.predict(
                point_coords=None,
                point_labels=None,
                # boxes=transformed_boxes,
                box=bounding_boxes,
                multimask_output=False,
            )
            if masks.ndim == 3:
                masks = torch.Tensor(masks).bool()
            else:
                masks = torch.Tensor(masks[:, 0, :, :]).bool()

            # Debug code, visualize all bounding boxes and segmentation masks

            image_vis = np.ascontiguousarray(np.array(rgb.permute(1, 2, 0)).astype(np.uint8))
            segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            # cv2.imwrite('clean_' + str(self.obs_count) + '.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            for idx, box in enumerate(bounding_boxes):
                tl_x, tl_y, br_x, br_y = box
                tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
                cv2.rectangle(
                    image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2
                )
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            for vis_mask in masks:
                segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
            image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            if not self.rerun:
                cv2.imwrite(self.log + "/seg" + str(self.obs_count) + ".jpg", image_vis)
            # else:
            #     rr.log('Segmentation mask', rr.Image(image_vis[:, :, [2, 1, 0]]))

            crops = []
            if not self.siglip:
                for (box, mask) in zip(bounding_boxes, masks):
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(
                        self.clip_preprocess(
                            transforms.ToPILImage()(
                                rgb[
                                    :,
                                    max(int(tl_y), 0) : min(int(br_y), rgb.shape[1]),
                                    max(int(tl_x), 0) : min(int(br_x), rgb.shape[2]),
                                ]
                            )
                        )
                    )
                features = self.clip_model.encode_image(torch.stack(crops, dim=0).to(self.device))
            else:
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(
                        rgb[
                            :,
                            max(int(tl_y), 0) : min(int(br_y), rgb.shape[1]),
                            max(int(tl_x), 0) : min(int(br_x), rgb.shape[2]),
                        ]
                    )
                inputs = self.clip_preprocess(
                    images=crops, padding="max_length", return_tensors="pt"
                ).to(self.device)
                features = self.clip_model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1).cpu()

        for idx, (sam_mask, feature) in enumerate(zip(masks.cpu(), features.cpu())):
            valid_mask = torch.logical_and(~mask, sam_mask)
            valid_xyz = world_xyz[valid_mask]
            if valid_xyz.shape[0] == 0:
                continue
            feature = feature.repeat(valid_xyz.shape[0], 1)
            valid_rgb = rgb.permute(1, 2, 0)[valid_mask]
            self.add_to_voxel_pcd(valid_xyz, feature, valid_rgb)

    def add_to_voxel_pcd(self, valid_xyz, feature, valid_rgb, weights=None, threshold=0.95):
        # Adding all points to voxelizedPointCloud is useless and expensive, we should exclude threshold of all points
        selected_indices = torch.randperm(len(valid_xyz))[: int((1 - threshold) * len(valid_xyz))]
        if len(selected_indices) == 0:
            return
        if valid_xyz is not None:
            valid_xyz = valid_xyz[selected_indices]
        if feature is not None:
            feature = feature[selected_indices]
        if valid_rgb is not None:
            valid_rgb = valid_rgb[selected_indices]
        if weights is not None:
            weights = weights[selected_indices]
        with self.voxel_map_lock:
            self.voxel_map_localizer.add(
                points=valid_xyz,
                features=feature,
                rgb=valid_rgb,
                weights=weights,
                obs_count=self.obs_count,
            )

    def process_rgbd_images(self, rgb, depth, intrinsics, pose):
        # import time
        # start_time = time.time()
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self.obs_count += 1

        cv2.imwrite(self.log + "/rgb" + str(self.obs_count) + ".jpg", rgb[:, :, [2, 1, 0]])
        np.save(self.log + "/rgb" + str(self.obs_count) + ".npy", rgb)
        np.save(self.log + "/depth" + str(self.obs_count) + ".npy", depth)
        np.save(self.log + "/intrinsics" + str(self.obs_count) + ".npy", intrinsics)
        np.save(self.log + "/pose" + str(self.obs_count) + ".npy", pose)

        rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth)
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)

        if self.image_shape is not None:
            h, w = self.image_shape
            h_image, w_image = depth.shape
            rgb = F.interpolate(
                rgb.unsqueeze(0), size=self.image_shape, mode="bilinear", align_corners=False
            ).squeeze()
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            intrinsics[0, 0] *= w / w_image
            intrinsics[1, 1] *= h / h_image
            intrinsics[0, 2] *= w / w_image
            intrinsics[1, 2] *= h / h_image

        world_xyz = get_xyz(depth, pose, intrinsics).squeeze()

        median_depth = torch.from_numpy(scipy.ndimage.median_filter(depth, size=5))
        median_filter_error = (depth - median_depth).abs()
        # edges = get_edges(depth, threshold=0.5)
        valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
        valid_depth = (
            valid_depth
            & (median_filter_error < 0.01).bool()
            # & ~edges
        )

        with self.voxel_map_lock:
            if self.vision_method == "mask&*lip":
                min_samples_clear = 3
            else:
                min_samples_clear = 10
            self.voxel_map_localizer.voxel_pcd.clear_points(
                depth,
                torch.from_numpy(intrinsics),
                torch.from_numpy(pose),
                min_samples_clear=min_samples_clear,
            )
            self.voxel_map.voxel_pcd.clear_points(
                depth, torch.from_numpy(intrinsics), torch.from_numpy(pose)
            )

        if self.vision_method == "mask&*lip":
            self.run_owl_sam_clip(rgb, ~valid_depth, world_xyz)
        else:
            self.run_mask_clip(rgb, ~valid_depth, world_xyz)

        self.voxel_map.add(
            camera_pose=torch.Tensor(pose),
            rgb=torch.Tensor(rgb).permute(1, 2, 0),
            depth=torch.Tensor(depth),
            camera_K=torch.Tensor(intrinsics),
        )
        obs, exp = self.voxel_map.get_2d_map()
        if self.rerun:
            # if not self.static:
            #     rr.set_time_sequence("frame", self.obs_count)
            # rr.log('robot_pov', rr.Image(rgb.permute(1, 2, 0)), static = self.static)
            if self.voxel_map.voxel_pcd._points is not None:
                rr.log(
                    "Obstalce_map/pointcloud",
                    rr.Points3D(
                        self.voxel_map.voxel_pcd._points.detach().cpu(),
                        colors=self.voxel_map.voxel_pcd._rgb.detach().cpu() / 255.0,
                        radii=0.03,
                    ),
                    static=self.static,
                )
            if self.voxel_map_localizer.voxel_pcd._points is not None:
                rr.log(
                    "Semantic_memory/pointcloud",
                    rr.Points3D(
                        self.voxel_map_localizer.voxel_pcd._points.detach().cpu(),
                        colors=self.voxel_map_localizer.voxel_pcd._rgb.detach().cpu() / 255.0,
                        radii=0.03,
                    ),
                    static=self.static,
                )
            # rr.log("Obstalce_map/2D_obs_map", rr.Image(obs.int() * 127 + exp.int() * 127))
        else:
            cv2.imwrite(
                self.log + "/debug_" + str(self.obs_count) + ".jpg",
                np.asarray(obs.int() * 127 + exp.int() * 127),
            )
        # end_time = time.time()
        # print('image processing takes ' + str(end_time - start_time) + ' seconds.')

    def read_from_pickle(self, pickle_file_name, num_frames: int = -1):
        print("Reading from ", pickle_file_name)
        rr.init("Debug", spawn=True)
        if isinstance(pickle_file_name, str):
            pickle_file_name = Path(pickle_file_name)
        assert pickle_file_name.exists(), f"No file found at {pickle_file_name}"
        with pickle_file_name.open("rb") as f:
            data = pickle.load(f)
        for i, (camera_pose, xyz, rgb, feats, depth, base_pose, K, world_xyz,) in enumerate(
            zip(
                data["camera_poses"],
                data["xyz"],
                data["rgb"],
                data["feats"],
                data["depth"],
                data["base_poses"],
                data["camera_K"],
                data["world_xyz"],
            )
        ):
            # Handle the case where we dont actually want to load everything
            if num_frames > 0 and i >= num_frames:
                break

            camera_pose = self.voxel_map.fix_data_type(camera_pose)
            xyz = self.voxel_map.fix_data_type(xyz)
            rgb = self.voxel_map.fix_data_type(rgb)
            depth = self.voxel_map.fix_data_type(depth)
            intrinsics = self.voxel_map.fix_data_type(K)
            if feats is not None:
                feats = self.voxel_map.fix_data_type(feats)
            base_pose = self.voxel_map.fix_data_type(base_pose)
            self.voxel_map.voxel_pcd.clear_points(depth, intrinsics, camera_pose)
            self.voxel_map.add(
                camera_pose=camera_pose,
                xyz=xyz,
                rgb=rgb,
                feats=feats,
                depth=depth,
                base_pose=base_pose,
                camera_K=K,
            )
            # points = self.voxel_map.voxel_pcd._points.detach().cpu()
            # rgb = self.voxel_map.voxel_pcd._rgb.detach().cpu() / 255.
            # rr.log("Obstalce_map/pointcloud", rr.Points3D(points[points[:, 2] > 0.1], \
            #                                                   colors=rgb[points[:, 2] > 0.1], radii=0.03))
            # plt.clf()
            # obstacles, _ = self.voxel_map.get_2d_map()
            # plt.imshow(obstacles)
            # grid_coord = self.voxel_map.xy_to_grid_coords(camera_pose[:2, 3])
            # plt.scatter(grid_coord[0], grid_coord[1], color = 'r')
            # buf = BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)
            # img = Image.open(buf)
            # img = np.array(img)
            # buf.close()
            # rr.log('2d_map', rr.Image(img))
            # rr.log('robot_point', rr.Points3D(camera_pose[:3, 3], colors=[1, 0, 0], radii=0.2))

            self.obs_count += 1
        self.voxel_map_localizer.voxel_pcd._points = data["combined_xyz"]
        self.voxel_map_localizer.voxel_pcd._features = data["combined_feats"]
        self.voxel_map_localizer.voxel_pcd._weights = data["combined_weights"]
        self.voxel_map_localizer.voxel_pcd._rgb = data["combined_rgb"]
        self.voxel_map_localizer.voxel_pcd._obs_counts = data["obs_id"]
        self.voxel_map_localizer.voxel_pcd._entity_ids = data["entity_id"]
        self.voxel_map_localizer.voxel_pcd.obs_count = max(
            self.voxel_map_localizer.voxel_pcd._obs_counts
        ).item()
        self.voxel_map.voxel_pcd.obs_count = max(
            self.voxel_map_localizer.voxel_pcd._obs_counts
        ).item()

    def write_to_pickle(self):
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
        if not os.path.exists("debug"):
            os.mkdir("debug")
        filename = "debug/" + self.log + ".pkl"
        data = {}
        data["camera_poses"] = []
        data["camera_K"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["world_xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        for frame in self.voxel_map.observations:
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["camera_K"].append(frame.camera_K)
            data["xyz"].append(frame.xyz)
            data["world_xyz"].append(frame.full_world_xyz)
            data["rgb"].append(frame.rgb)
            data["depth"].append(frame.depth)
            data["feats"].append(frame.feats)
            for k, v in frame.info.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        (
            data["combined_xyz"],
            data["combined_feats"],
            data["combined_weights"],
            data["combined_rgb"],
        ) = self.voxel_map_localizer.voxel_pcd.get_pointcloud()
        data["obs_id"] = self.voxel_map_localizer.voxel_pcd._obs_counts
        data["entity_id"] = self.voxel_map_localizer.voxel_pcd._entity_ids
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("write all data to", filename)
