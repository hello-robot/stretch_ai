# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter, median_filter
from torch import Tensor

from stretch.llms import OpenaiClient
from stretch.llms.prompts import DYNAMEM_VISUAL_GROUNDING_PROMPT
from stretch.llms.qwen_client import Qwen25VLClient
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.utils.image import Camera, camera_xyz_to_global_xyz
from stretch.utils.morphology import binary_dilation, binary_erosion, get_edges
from stretch.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from stretch.utils.voxel import VoxelizedPointcloud, scatter3d

from .voxel import VALID_FRAMES, Frame
from .voxel import SparseVoxelMap as SparseVoxelMapBase

logger = logging.getLogger(__name__)


class SparseVoxelMap(SparseVoxelMapBase):
    def __init__(
        self,
        resolution: float = 0.01,
        semantic_memory_resolution: float = 0.05,
        feature_dim: int = 3,
        grid_size: Tuple[int, int] = None,
        grid_resolution: float = 0.05,
        obs_min_height: float = 0.1,
        obs_max_height: float = 1.8,
        obs_min_density: float = 10,
        smooth_kernel_size: int = 2,
        neg_obs_height: float = 0.0,
        add_local_radius_points: bool = True,
        remove_visited_from_obstacles: bool = False,
        local_radius: float = 0.8,
        min_depth: float = 0.25,
        max_depth: float = 2.5,
        pad_obstacles: int = 0,
        background_instance_label: int = -1,
        instance_memory_kwargs: Dict[str, Any] = {},
        voxel_kwargs: Dict[str, Any] = {},
        encoder: Optional[MaskSiglipEncoder] = None,
        map_2d_device: str = "cpu",
        device: Optional[str] = None,
        use_instance_memory: bool = False,
        use_median_filter: bool = False,
        median_filter_size: int = 5,
        median_filter_max_error: float = 0.01,
        use_derivative_filter: bool = False,
        derivative_filter_threshold: float = 0.5,
        prune_detected_objects: bool = False,
        add_local_radius_every_step: bool = False,
        min_points_per_voxel: int = 10,
        use_negative_obstacles: bool = False,
        point_update_threshold: float = 0.9,
        detection=None,
        image_shape=(480, 360),
        log="test",
        mllm=False,
        run_eqa=False,
    ):
        super().__init__(
            resolution=resolution,
            feature_dim=feature_dim,
            grid_size=grid_size,
            grid_resolution=grid_resolution,
            obs_min_height=obs_min_height,
            obs_max_height=obs_max_height,
            obs_min_density=obs_min_density,
            smooth_kernel_size=smooth_kernel_size,
            neg_obs_height=neg_obs_height,
            add_local_radius_points=add_local_radius_points,
            remove_visited_from_obstacles=remove_visited_from_obstacles,
            local_radius=local_radius,
            min_depth=min_depth,
            max_depth=max_depth,
            pad_obstacles=pad_obstacles,
            background_instance_label=background_instance_label,
            instance_memory_kwargs=instance_memory_kwargs,
            voxel_kwargs=voxel_kwargs,
            encoder=encoder,
            map_2d_device=map_2d_device,
            device=device,
            use_instance_memory=use_instance_memory,
            use_median_filter=use_median_filter,
            median_filter_size=median_filter_size,
            median_filter_max_error=median_filter_max_error,
            use_derivative_filter=use_derivative_filter,
            derivative_filter_threshold=derivative_filter_threshold,
            prune_detected_objects=prune_detected_objects,
            add_local_radius_every_step=add_local_radius_every_step,
            min_points_per_voxel=min_points_per_voxel,
            use_negative_obstacles=use_negative_obstacles,
        )

        self.point_update_threshold = point_update_threshold
        self._history_soft: Optional[Tensor] = None
        self.semantic_memory = VoxelizedPointcloud(voxel_size=semantic_memory_resolution).to(
            self.device
        )
        self.encoder = encoder
        self.image_shape = image_shape
        self.obs_count = 0
        self.detection_model = detection
        self.log = log
        self.mllm = mllm
        if self.mllm:
            # Used to do visual grounding task
            self.gpt_client = OpenaiClient(
                DYNAMEM_VISUAL_GROUNDING_PROMPT, model="gpt-4o-2024-05-13"
            )

        self.run_eqa = run_eqa
        if self.run_eqa:
            # To avoid using too much closed source VLMs, we use Qwen2.5-3b-vl-instruct for image description.
            self.image_description_client = Qwen25VLClient(
                model_size="3B", quantization="int4", max_tokens=20
            )

            self.image_descriptions: List[Tuple[List[str], List[int]]] = []

            from stretch.llms.gemini_client import GeminiClient
            from stretch.llms.prompts.eqa_prompt import EQA_PROMPT

            self.eqa_client = GeminiClient(EQA_PROMPT, model="gemini-2.5-pro-preview-03-25")

        # Attributes for EQA, If you are not running EQA module, this will stay the same.
        self._question: Optional[str] = None
        self.relevant_objects: Optional[list] = None

        self.history_outputs: List[str] = []

    def find_alignment_over_model(self, queries: str):
        clip_text_tokens = self.encoder.encode_text(queries).cpu()
        points, features, weights, _ = self.semantic_memory.get_pointcloud()
        if points is None:
            return None
        features = F.normalize(features, p=2, dim=-1).cpu()
        point_alignments = clip_text_tokens.float() @ features.float().T

        # print(point_alignments.shape)
        return point_alignments

    def find_alignment_for_text(self, text: str):
        points, features, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        return points[alignments.argmax(dim=-1)].detach().cpu()

    def find_obs_id_for_text(self, text: str):
        obs_counts = self.semantic_memory._obs_counts
        alignments = self.find_alignment_over_model(text).cpu()
        return obs_counts[alignments.argmax(dim=-1)].detach().cpu()

    def verify_point(
        self,
        text: str,
        point: Union[torch.Tensor, np.ndarray],
        distance_threshold: float = 0.1,
        similarity_threshold: float = 0.14,
    ):
        """
        Running visual grounding is quite time consuming.
        Thus, sometimes if the point has very high cosine similarity with text query, we might opt not to run visual grounding again.
        This function evaluates the cosine similarity.
        """
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        distances = torch.linalg.norm(point - points.detach().cpu(), dim=-1)
        if torch.min(distances) > distance_threshold:
            print("Points are so far from other points!")
            return False
        alignments = self.find_alignment_over_model(text).detach().cpu()[0]
        if torch.max(alignments[distances <= distance_threshold]) < similarity_threshold:
            print("Points close the the point are not similar to the text!")
        return torch.max(alignments[distances < distance_threshold]) >= similarity_threshold

    def get_2d_map(
        self, debug: bool = False, return_history_id: bool = False, kernel: int = 7
    ) -> Tuple[Tensor, ...]:
        """
        Get 2d map with explored area and frontiers.
        return_history_id: if True, return when each voxel was recently updated
        """

        # Is this already cached? If so we don't need to go to all this work
        if (
            self._map2d is not None
            and self._history_soft is not None
            and self._seq == self._2d_last_updated
        ):
            return self._map2d if not return_history_id else (*self._map2d, self._history_soft)

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz, _, counts, _ = self.voxel_pcd.get_pointcloud()
        # print(counts)
        # if xyz is not None:
        #     counts = torch.ones(xyz.shape[0])
        obs_ids = self.voxel_pcd._obs_counts
        if xyz is None:
            xyz = torch.zeros((0, 3))
            counts = torch.zeros((0))
            obs_ids = torch.zeros((0))

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        # print('min_height', min_height, 'max_height', max_height)
        grid_size = self.grid_size + [max_height]
        voxels = torch.zeros(grid_size, device=device)

        # Mask out obstacles only above a certain height
        obs_mask = xyz[:, -1] < max_height
        xyz = xyz[obs_mask, :]
        counts = counts[obs_mask][:, None]
        # print(counts)
        obs_ids = obs_ids[obs_mask][:, None]

        # voxels[x_coords, y_coords, z_coords] = 1
        voxels = scatter3d(xyz, counts, grid_size)
        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:max_height]
        obstacles_soft = torch.sum(obstacle_voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density

        history_ids = history_ids[:, :, min_height:max_height]
        history_soft = torch.max(history_ids, dim=-1).values
        history_soft = torch.from_numpy(maximum_filter(history_soft.float().numpy(), size=kernel))

        if self._remove_visited_from_obstacles:
            # Remove "visited" points containing observations of the robot
            obstacles *= (1 - self._visited).bool()

        if self.dilate_obstacles_kernel is not None:
            obstacles = binary_dilation(
                obstacles.float().unsqueeze(0).unsqueeze(0),
                self.dilate_obstacles_kernel,
            )[0, 0].bool()

        # Explored area = only floor mass
        # floor_voxels = voxels[:, :, :min_height]
        explored_soft = torch.sum(voxels, dim=-1)

        # Add explored radius around the robot, up to min depth
        explored = explored_soft > 0
        explored = (torch.zeros_like(explored) + self._visited).to(torch.bool) | explored

        if self.smooth_kernel_size > 0:
            # Opening and closing operations here on explore
            explored = binary_erosion(
                binary_dilation(explored.float().unsqueeze(0).unsqueeze(0), self.smooth_kernel),
                self.smooth_kernel,
            )
            explored = binary_dilation(
                binary_erosion(explored, self.smooth_kernel),
                self.smooth_kernel,
            )[0, 0].bool()
        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(2, 2, 1)
            plt.imshow(obstacles_soft.detach().cpu().numpy())
            plt.title("obstacles soft")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(explored_soft.detach().cpu().numpy())
            plt.title("explored soft")
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(obstacles.detach().cpu().numpy())
            plt.title("obstacles")
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(explored.detach().cpu().numpy())
            plt.axis("off")
            plt.title("explored")
            plt.show()

        # Set the boundary in case the robot runs out from the environment
        obstacles[0:30, :] = True
        obstacles[-30:, :] = True
        obstacles[:, 0:30] = True
        obstacles[:, -30:] = True
        # Generate exploration heuristic to prevent robot from staying around the boundary
        if history_soft is not None:
            history_soft[0:35, :] = history_soft.max().item()
            history_soft[-35:, :] = history_soft.max().item()
            history_soft[:, 0:35] = history_soft.max().item()
            history_soft[:, -35:] = history_soft.max().item()

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        self._history_soft = history_soft
        if not return_history_id:
            return obstacles, explored
        else:
            return obstacles, explored, history_soft

    def process_rgbd_images(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray
    ):
        """
        Process rgbd images for Dynamem
        """
        # Log input data
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self.obs_count += 1

        cv2.imwrite(self.log + "/rgb" + str(self.obs_count) + ".jpg", rgb[:, :, [2, 1, 0]])
        np.save(self.log + "/rgb" + str(self.obs_count) + ".npy", rgb)
        np.save(self.log + "/depth" + str(self.obs_count) + ".npy", depth)
        np.save(self.log + "/intrinsics" + str(self.obs_count) + ".npy", intrinsics)
        np.save(self.log + "/pose" + str(self.obs_count) + ".npy", pose)

        # Update obstacle map
        self.voxel_pcd.clear_points(
            torch.from_numpy(depth), torch.from_numpy(intrinsics), torch.from_numpy(pose)
        )
        self.add(
            camera_pose=torch.Tensor(pose),
            rgb=torch.Tensor(rgb),
            depth=torch.Tensor(depth),
            camera_K=torch.Tensor(intrinsics),
        )

        # Add image descriptions if we want to explore intelligently
        if self.run_eqa:
            self.list_objects_in_an_image(rgb)

        # Process data: reshaping images, computing xyz coordinate, depth filtering
        rgb, depth = torch.Tensor(rgb), torch.Tensor(depth)
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)

        if self.image_shape is not None:
            h, w = self.image_shape
            h_image, w_image = depth.shape
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            intrinsics = np.copy(intrinsics)
            intrinsics[0, 0] *= w / w_image
            intrinsics[1, 1] *= h / h_image
            intrinsics[0, 2] *= w / w_image
            intrinsics[1, 2] *= h / h_image

        height, width = depth.squeeze().shape
        camera = Camera.from_K(np.array(intrinsics), width=width, height=height)
        camera_xyz = camera.depth_to_xyz(np.array(depth))
        world_xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(pose)))

        median_depth = torch.from_numpy(median_filter(depth, size=5))
        median_filter_error = (depth - median_depth).abs()
        valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
        valid_depth = valid_depth & (median_filter_error < 0.01).bool()
        mask = ~valid_depth

        # Update semantic memory
        self.semantic_memory.clear_points(
            depth, torch.from_numpy(intrinsics), torch.from_numpy(pose), min_samples_clear=10
        )

        with torch.no_grad():
            rgb, features = self.encoder.run_mask_siglip(rgb, self.image_shape)  # type:ignore
            rgb, features = rgb.squeeze(), features.squeeze()

        valid_xyz = world_xyz[~mask]
        features = features[~mask]
        valid_rgb = rgb.permute(1, 2, 0)[~mask]
        if len(valid_xyz) != 0:
            self.add_to_semantic_memory(valid_xyz, features, valid_rgb)

    def add_to_semantic_memory(
        self,
        valid_xyz: Optional[torch.Tensor],
        feature: Optional[torch.Tensor],
        valid_rgb: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
        threshold: float = 0.95,
    ):
        """
        Add pixel points into the semantic memory
        """
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

        valid_xyz = valid_xyz.to(self.device)
        if feature is not None:
            feature = feature.to(self.device)
        if valid_rgb is not None:
            valid_rgb = valid_rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        self.semantic_memory.add(
            points=valid_xyz,
            features=feature,
            rgb=valid_rgb,
            weights=weights,
            obs_count=self.obs_count,
        )

    def localize_text(self, text, debug=True, return_debug=False):
        if self.mllm:
            return self.localize_with_mllm(text, debug=debug, return_debug=return_debug)
        else:
            return self.localize_with_feature_similarity(
                text, debug=debug, return_debug=return_debug
            )

    def find_all_images(
        self,
        text: str,
        min_similarity_threshold: Optional[float] = None,
        min_point_num: int = 100,
        max_img_num: Optional[int] = 3,
    ):
        """
        Select all images with high pixel similarity with text (by identifying whether points in this image are relevant objects)

        Args:
            min_similarity_threshold: Make sure every point with similarity greater than this value would be considered as the relevant objects
            min_point_num: Make sure we select at least these many points as relevant images.
            max_img_num: The maximum number of images we want to identify as relevant objects.
        """
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        points = points.cpu()
        alignments = self.find_alignment_over_model(text).cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts.cpu()

        turning_point = (
            min(min_similarity_threshold, alignments[torch.argsort(alignments)[-min_point_num]])
            if min_similarity_threshold is not None
            else alignments[torch.argsort(alignments)[-min_point_num]]
        )
        mask = alignments >= turning_point
        obs_counts = obs_counts[mask]
        alignments = alignments[mask]
        points = points[mask]

        unique_obs_counts, inverse_indices = torch.unique(obs_counts, return_inverse=True)

        points_with_max_alignment = torch.zeros((len(unique_obs_counts), points.size(1)))
        max_alignments = torch.zeros(len(unique_obs_counts))

        for i in range(len(unique_obs_counts)):
            # Get indices of elements belonging to the current cluster
            indices_in_cluster = (inverse_indices == i).nonzero(as_tuple=True)[0]
            if len(indices_in_cluster) <= 2:
                continue

            # Extract the alignments and points for the current cluster
            cluster_alignments = alignments[indices_in_cluster].squeeze()
            cluster_points = points[indices_in_cluster]

            # Find the point with the highest alignment in the cluster
            max_alignment_idx_in_cluster = cluster_alignments.argmax()
            point_with_max_alignment = cluster_points[max_alignment_idx_in_cluster]

            # Store the result
            points_with_max_alignment[i] = point_with_max_alignment
            max_alignments[i] = cluster_alignments.max()

        if max_img_num is not None:
            top_k = min(max_img_num, len(max_alignments))
        else:
            top_k = len(max_alignments)
        top_alignments, top_indices = torch.topk(
            max_alignments, k=top_k, dim=0, largest=True, sorted=True
        )
        top_points = points_with_max_alignment[top_indices]
        top_obs_counts = unique_obs_counts[top_indices]

        sorted_obs_counts, sorted_indices = torch.sort(top_obs_counts, descending=False)
        sorted_points = top_points[sorted_indices]
        top_alignments = top_alignments[sorted_indices]

        return sorted_obs_counts, sorted_points, top_alignments

    def llm_locator(self, image_ids: Union[torch.Tensor, np.ndarray, list], text: str):
        """
        Prompting the mLLM to select the images containing objects of interest.

        Input:
            image_ids: a series of images you want to send to mLLM
            text: text query

        Return
        """
        user_messages = []
        for obs_id in image_ids:
            obs_id = int(obs_id) - 1
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            depth = self.observations[obs_id].depth
            rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            user_messages.append(image)
        user_messages.append("The object you need to find is " + text)

        response = self.gpt_client(user_messages)
        return self.parse_localization_response(response)

    def parse_localization_response(self, response: str):
        """
        Parse the output of GPT4o to extract the selected image's id
        """
        try:
            # Use regex to locate the 'Images:' section, allowing for varying whitespace and line breaks
            images_section_match = re.search(r"Images:\s*([\s\S]+)", response, re.IGNORECASE)
            if not images_section_match:
                raise ValueError("The 'Images:' section is missing.")

            # Extract the content after 'Images:'
            images_content = images_section_match.group(1).strip()

            # Check if the content is 'None' (case-insensitive)
            if images_content.lower() == "none":
                return None

            # Use regex to find all numbers, regardless of separators like commas, periods, or spaces
            numbers = re.findall(r"\d+", images_content)

            if not numbers:
                raise ValueError("No numbers found in the 'Images:' section.")

            # Convert all found numbers to integers
            numbers = [int(num) for num in numbers]

            # Return all numbers as a list if multiple numbers are found
            if len(numbers) > 0:
                return numbers[-1]
            else:
                return None

        except Exception as e:
            # Handle any exceptions and optionally log the error message
            print(f"Error: {e}")
            return None

    def localize_with_mllm(self, text: str, debug=True, return_debug=False):
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts
        image_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        debug_text = ""
        target_point = None

        image_ids, points, alignments = self.find_all_images(
            # text, min_similarity_threshold=0.12, max_img_num=3
            text,
            max_img_num=3,
        )
        target_id = self.llm_locator(image_ids, text)

        if target_id is None:
            debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
            image_id = None
            point = None
        else:
            target_id -= 1
            target_point = points[target_id]
            image_id = image_ids[target_id]
            point = points[target_id]
            debug_text += "#### - An image is identified \n"

        if image_id is not None:
            rgb = self.observations[image_id - 1].rgb
            pose = self.observations[image_id - 1].camera_pose
            depth = self.observations[image_id - 1].depth
            K = self.observations[image_id - 1].camera_K

            res = self.detection_model.compute_obj_coord(text, rgb, depth, K, pose)
            if res is not None:
                target_point = res
            else:
                target_point = point

        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, image_id, point

    def localize_with_feature_similarity(self, text, debug=True, return_debug=False):
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts
        obs_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        debug_text = ""
        target_point = None

        if obs_id <= 0 or obs_id > len(self.observations):
            res = None
        else:
            rgb = self.observations[obs_id - 1].rgb
            pose = self.observations[obs_id - 1].camera_pose
            depth = self.observations[obs_id - 1].depth
            K = self.observations[obs_id - 1].camera_K

            res = self.detection_model.compute_obj_coord(text, rgb, depth, K, pose)

        if res is not None:
            target_point = res
            debug_text += (
                "#### - Object is detected in observations . **😃** Directly navigate to it.\n"
            )
        else:
            # debug_text += '#### - Directly ignore this instance is the target. **😞** \n'
            cosine_similarity_check = alignments.max().item() > 0.14
            if cosine_similarity_check:
                target_point = point

                debug_text += (
                    "#### - The point has high cosine similarity. **😃** Directly navigate to it.\n"
                )
            else:
                debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
        # print('target_point', target_point)
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs_id, point

    def add(
        self,
        camera_pose: Tensor,
        rgb: Tensor,
        xyz: Optional[Tensor] = None,
        camera_K: Optional[Tensor] = None,
        feats: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        base_pose: Optional[Tensor] = None,
        xyz_frame: str = "camera",
        **info,
    ):
        """Add this to our history of observations. Also update the current running map.

        Parameters:
            camera_pose(Tensor): [4,4] cam_to_world matrix
            rgb(Tensor): N x 3 color points
            camera_K(Tensor): [3,3] camera instrinsics matrix -- usually pinhole model
            xyz(Tensor): N x 3 point cloud points in camera coordinates
            feats(Tensor): N x D point cloud features; D == 3 for RGB is most common
            base_pose(Tensor): optional location of robot base
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        # Shape checking
        assert rgb.ndim == 3 or rgb.ndim == 2, f"{rgb.ndim=}: must be 2 or 3"
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        if isinstance(camera_pose, np.ndarray):
            camera_pose = torch.from_numpy(camera_pose)
        if depth is not None:
            assert (
                rgb.shape[:-1] == depth.shape
            ), f"depth and rgb image sizes must match; got {rgb.shape=} {depth.shape=}"
        assert xyz is not None or (camera_K is not None and depth is not None)
        if xyz is not None:
            assert (
                xyz.shape[-1] == 3
            ), "xyz must have last dimension = 3 for x, y, z position of points"
            assert rgb.shape == xyz.shape, "rgb shape must match xyz"
            # Make sure shape is correct here for xyz and any passed-in features
            if feats is not None:
                assert (
                    feats.shape[-1] == self.feature_dim
                ), f"features must match voxel feature dimenstionality of {self.feature_dim}"
                assert xyz.shape[0] == feats.shape[0], "features must be available for each point"
            else:
                pass
            if isinstance(xyz, np.ndarray):
                xyz = torch.from_numpy(xyz)
        if depth is not None:
            assert depth.ndim == 2 or xyz_frame == "world"
        if camera_K is not None:
            assert camera_K.ndim == 2, "camera intrinsics K must be a 3x3 matrix"
        assert (
            camera_pose.ndim == 2 and camera_pose.shape[0] == 4 and camera_pose.shape[1] == 4
        ), "Camera pose must be a 4x4 matrix representing a pose in SE(3)"
        assert (
            xyz_frame in VALID_FRAMES
        ), f"frame {xyz_frame} was not valid; should one one of {VALID_FRAMES}"

        # Apply a median filter to remove bad depth values when mapping and exploring
        # This is not strictly necessary but the idea is to clean up bad pixels
        if depth is not None and self.use_median_filter:
            median_depth = torch.from_numpy(median_filter(depth, size=self.median_filter_size))
            median_filter_error = (depth - median_depth).abs()

        # Get full_world_xyz
        if xyz is not None:
            if xyz_frame == "camera":
                full_world_xyz = (
                    torch.cat([xyz, torch.ones_like(xyz[..., [0]])], dim=-1) @ camera_pose.T
                )[..., :3]
            elif xyz_frame == "world":
                full_world_xyz = xyz
            else:
                raise NotImplementedError(f"Unknown xyz_frame {xyz_frame}")
        else:
            full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
                depth=depth.unsqueeze(0).unsqueeze(1),
                pose=camera_pose.unsqueeze(0),
                inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            )
        # add observations before we start changing things
        self.observations.append(
            Frame(
                camera_pose,
                camera_K,
                xyz,
                rgb,
                feats,
                depth,
                instance=None,
                instance_classes=None,
                instance_scores=None,
                base_pose=base_pose,
                info=info,
                obs=None,
                full_world_xyz=full_world_xyz,
                xyz_frame=xyz_frame,
            )
        )

        valid_depth = torch.full_like(rgb[:, 0], fill_value=True, dtype=torch.bool)
        if depth is not None:
            valid_depth = (depth > self.min_depth) & (depth < self.max_depth)

            if self.use_derivative_filter:
                edges = get_edges(depth, threshold=self.derivative_filter_threshold)
                valid_depth = valid_depth & ~edges

            if self.use_median_filter:
                valid_depth = (
                    valid_depth & (median_filter_error < self.median_filter_max_error).bool()
                )

        # Add to voxel grid
        if feats is not None:
            feats = feats[valid_depth].reshape(-1, feats.shape[-1])
        rgb = rgb[valid_depth].reshape(-1, 3)
        world_xyz = full_world_xyz.view(-1, 3)[valid_depth.flatten()]

        # TODO: weights could also be confidence, inv distance from camera, etc
        if world_xyz.nelement() > 0:
            selected_indices = torch.randperm(len(world_xyz))[
                : int((1 - self.point_update_threshold) * len(world_xyz))
            ]
            if len(selected_indices) == 0:
                return
            if world_xyz is not None:
                world_xyz = world_xyz[selected_indices]
            if feats is not None:
                feats = feats[selected_indices]
            if rgb is not None:
                rgb = rgb[selected_indices]
            self.voxel_pcd.add(world_xyz, features=feats, rgb=rgb, weights=None)

        if self._add_local_radius_points:
            # TODO: just get this from camera_pose?
            self._update_visited(camera_pose[:3, 3].to(self.map_2d_device))
        if base_pose is not None:
            self._update_visited(base_pose.to(self.map_2d_device))

        # Increment sequence counter
        self._seq += 1

    def xy_to_grid_coords(self, xy: np.ndarray) -> Optional[np.ndarray]:
        if not isinstance(xy, np.ndarray):
            xy = np.array(xy)
        return self.grid.xy_to_grid_coords(torch.Tensor(xy))

    def grid_coords_to_xy(self, grid_coords: np.ndarray) -> np.ndarray:
        if not isinstance(grid_coords, np.ndarray):
            grid_coords = np.array(grid_coords)
        return self.grid.grid_coords_to_xy(torch.Tensor(grid_coords))

    def grid_coords_to_xyt(self, grid_coords: np.ndarray) -> np.ndarray:
        if not isinstance(grid_coords, np.ndarray):
            grid_coords = np.array(grid_coords)
        return self.grid.grid_coords_to_xyt(torch.Tensor(grid_coords))

    def read_from_pickle(self, pickle_file_name, num_frames: int = -1):
        print("Reading from ", pickle_file_name)
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

            camera_pose = self.fix_data_type(camera_pose)
            xyz = self.fix_data_type(xyz)
            rgb = self.fix_data_type(rgb)
            depth = self.fix_data_type(depth)
            intrinsics = self.fix_data_type(K)
            if feats is not None:
                feats = self.fix_data_type(feats)
            base_pose = self.fix_data_type(base_pose)
            self.voxel_pcd.clear_points(depth, intrinsics, camera_pose)
            self.add(
                camera_pose=camera_pose,
                xyz=xyz,
                rgb=rgb,
                feats=feats,
                depth=depth,
                base_pose=base_pose,
                camera_K=K,
            )

            self.obs_count += 1
        self.semantic_memory._points = data["combined_xyz"]
        self.semantic_memory._features = data["combined_feats"]
        self.semantic_memory._weights = data["combined_weights"]
        self.semantic_memory._rgb = data["combined_rgb"]
        self.semantic_memory._obs_counts = data["obs_id"]
        self.semantic_memory._mins = self.semantic_memory._points.min(dim=0).values
        self.semantic_memory._maxs = self.semantic_memory._points.max(dim=0).values
        self.semantic_memory.obs_count = max(self.semantic_memory._obs_counts).item()
        self.semantic_memory.obs_count = max(self.semantic_memory._obs_counts).item()

    def write_to_pickle(self, filename: Optional[str] = None) -> None:
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts.

        Args:
            filename (Optional[str], optional): Filename to write to. Defaults to None.
        """
        if not os.path.exists("debug"):
            os.mkdir("debug")
        if filename is None:
            filename = self.log + ".pkl"
        data: Dict[str, Any] = {}
        data["camera_poses"] = []
        data["camera_K"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["world_xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        for frame in self.observations:
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
        ) = self.semantic_memory.get_pointcloud()
        data["obs_id"] = self.semantic_memory._obs_counts
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("write all data to", filename)

    ##################################################
    #
    # Most EQA module codes start from here
    #
    ###################################################

    def get_active_image_descriptions(self):
        """
        Return a list of image descriptions that are still active. By active it means there is still some voxel in voxel map associated with it.
        """
        if self.voxel_pcd._points is None:
            return None

        # Extract image id for each 2d grid points
        obs_ids = self.voxel_pcd._obs_counts
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        max_height = int(self.obs_max_height / self.grid_resolution)
        grid_size = self.grid_size + [max_height]
        obs_ids = obs_ids[:, None]

        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")
        history = torch.max(history_ids, dim=-1).values
        history = torch.from_numpy(maximum_filter(history.float().numpy(), size=5))
        history[0:35, :] = history.max().item()
        history[-35:, :] = history.max().item()
        history[:, 0:35] = history.max().item()
        history[:, -35:] = history.max().item()
        # from matplotlib import pyplot as plt
        # plt.imshow(history)
        # plt.show()

        selected_images = torch.unique(history).int()
        # history image id is 1-indexed, so we need to subtract 1 from scores
        return (
            history,
            selected_images,
            [
                self.image_descriptions[selected_image.item() - 1]
                for selected_image in selected_images
            ],
        )

    def extract_relevant_objects(self, question: str):
        """
        Parsed the question and extract few keywords for DynaMem voxel map to select relevant images
        """
        if self._question != question:
            self._question = question
            # The cached question is not the same as the question provided
            prompt = """
                Assume there is an agent doing Question Answering in an environment.
                When it receives a question, you need to tell the agent few objects (preferably 1-3) it needs to pay special attention to.
                Example:
                    Where is the pen?
                    pen

                    Is there grey cloth on cloth hanger?
                    gery cloth,cloth hanger
            """
            messages = [prompt, self._question]
            # To avoid initializing too many clients and using up too much memory, I reused the client generating the image descriptions even though it is a VL model
            self.relevant_objects = self.image_description_client(messages).split(",")
            print("relevant objects to look at", self.relevant_objects)
            self.history_outputs = []

    def log_text(self, commands):
        """
        Log the text input and image input into some files for debugging and visualization
        """
        if not os.path.exists(self.log + "/" + str(len(self.image_descriptions))):
            os.makedirs(self.log + "/" + str(len(self.image_descriptions)))
            input_texts = ""
            for command in commands:
                input_texts += command + "\n"
            with open(
                self.log + "/" + str(len(self.image_descriptions)) + "/input.txt", "w"
            ) as file:
                file.write(input_texts)

    def parse_answer(self, answer_outputs: str):

        """
        Parse the output of LLM text into reasoning, answer, confidence, action, confidence_reasoning
        """

        # Log LLM output
        with open(self.log + "/" + str(len(self.image_descriptions)) + "/output.txt", "w") as file:
            file.write(answer_outputs)

        # Answer outputs in the format "Caption: Reasoning: Answer: Confidence: Action: Confidence_reasoning:"
        def extract_between(text, start, end):
            try:
                return (
                    text.split(start, 1)[1]
                    .split(end, 1)[0]
                    .strip()
                    .replace("\n", "")
                    .replace("\t", "")
                )
            except IndexError:
                return ""

        def extract_after(text, start):
            try:
                return text.split(start, 1)[1].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        reasoning = extract_between(answer_outputs, "reasoning:", "answer:")
        answer = extract_between(answer_outputs, "answer:", "confidence:")
        confidence_text = extract_between(answer_outputs, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "")
        action = extract_between(answer_outputs, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(answer_outputs, "confidence_reasoning:")

        return reasoning, answer, confidence, action, confidence_reasoning

    def query_answer(self, question: str, xyt, planner):
        """
        Util function to prompt mLLM to provide answer output, and process the raw answer output into robot's next step.
        """

        # Extract keywords from the question
        self.extract_relevant_objects(question)

        # messages = [{"type": "text", "text": "Question: " + question}]
        commands: List[Any] = ["Question: " + question]
        # messages.append({"type": "text", "text": "HISTORY: "})
        commands.append("HISTORY: ")
        for (i, history_output) in enumerate(self.history_outputs):
            # messages.append({"type": "text", "text": "Iteration_" + str(i) + ":" + history_output})
            commands.append("Iteration_" + str(i) + ":" + history_output)
        # messages.append({"role": "user", "content": [{"type": "input_text", "text": question}]})

        # Select the task relevant images with DynaMem
        img_idx = 0
        all_obs_ids = set()

        for relevant_object in self.relevant_objects:
            # Limit the total number of images to 6
            image_ids, _, _ = self.find_all_images(
                relevant_object,
                min_similarity_threshold=0.12,
                max_img_num=6 // len(self.relevant_objects),
                min_point_num=40,
            )
            for obs_id in image_ids:
                obs_id = int(obs_id) - 1
                all_obs_ids.add(obs_id)

        all_obs_ids = list(all_obs_ids)  # type: ignore

        # Prepare the visual clues (image descriptions)
        selected_images, action_prompt = self.get_image_descriptions_str(xyt, planner, all_obs_ids)
        commands.append(action_prompt)
        self.log_text(commands)
        relevant_images = []

        for obs_id in all_obs_ids:
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Log the input images
            image.save(
                self.log + "/" + str(len(self.image_descriptions)) + "/" + str(img_idx) + ".jpg"
            )
            img_idx += 1

            commands.append(image)
            relevant_images.append(image)

        # Extract answers
        answer_outputs = (
            self.eqa_client(commands).replace("*", "").replace("/", "").replace("#", "").lower()
        )

        print(commands)
        print(answer_outputs)

        (
            reasoning,
            answer,
            confidence,
            action,
            confidence_reasoning,
        ) = self.parse_answer(answer_outputs)

        # If the robot is not confident, it should plan exploration
        if not confidence:
            action = selected_images[int(action) - 1]
            rgb = np.copy(self.observations[action - 1].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Cache conversations between the robot and the mLLM for the next iteration of question answering planning
            self.history_outputs.append(
                "Answer:"
                + answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction:"
                + "Navigate to Image with objects "
                + str(self.image_descriptions[action - 1][0])
                + " with grid coord "
                + str(self.image_descriptions[action - 1][1])
                + "\nConfidence reasoning:"
                + confidence_reasoning
            )
        else:
            action = None

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            self.get_target_point_from_image_id(action, xyt, planner)
            if action is not None
            else None,
            relevant_images,
        )

    def get_image_descriptions_str(self, xyt, planner, obs_ids):
        """
        Select visual clues of all active images (images still associated with some voxel points in the voxel map)
        """
        (
            _,
            selected_images,
            image_descriptions,
        ) = self.get_active_image_descriptions()
        frontier_ids = list(self.get_frontier_ids(xyt, planner))
        options = ""
        if len(image_descriptions) > 0:
            for i, (cluster, grid_coord) in enumerate(image_descriptions):
                index = selected_images[i]
                cluster_string = ""
                for ob in cluster:
                    cluster_string += ob + ", "
                cluster_string = cluster_string[:-2] + ";"
                # Indicate the grid coord this image describes to avoid redundant exploration.
                cluster_string += " This image is taken at grid coords " + str(grid_coord)
                # If we have already send the raw image observation to LLM.
                if index in obs_ids:
                    cluster_string += (
                        " This observation description is associated with Image "
                        + str(obs_ids.index(index) + 1)
                        + ";"
                    )
                # If this image corresponds to an unexplored frontier
                if index in frontier_ids:
                    cluster_string += (
                        " This observation description corresponds to unexplored space;"
                    )
                options += f"{i+1}. {cluster_string}\n"
        return selected_images, "IMAGE_DESCRIPTIONS: " + options

    def get_target_point_from_image_id(self, image_id: int, xyt, planner):
        """
        When the robot is not confident with the answer, mLLM will output an image id indicating a rough direction for the robot to take the next step.
        This function selects the target point's xy coordinate based on the image id provided.
        """

        # history output by get_active_descriptions output a history id map considering history id of the floor point
        # history_soft output by get_2d_map output a history id map excluding history id of the floor point
        # Therefore, history is generally used to select active image observations while history_soft is generally used to determine unexplored frontier
        (
            history,
            _,
            _,
        ) = self.get_active_image_descriptions()
        obstacles, explored = self.get_2d_map()
        outside_frontier = self.get_outside_frontier(xyt, planner)
        unexplored_frontier = outside_frontier & ~explored
        # Navigation priority: unexplored frontier > obstalces > others
        if torch.sum((history == image_id) & unexplored_frontier) > 0:
            print("unexplored frontier")
            image_coord = (
                ((history == image_id) & unexplored_frontier)
                .nonzero(as_tuple=False)
                .median(dim=0)
                .values.int()
            )
        elif torch.sum((history == image_id) & obstacles) > 0:
            print("obstacles")
            image_coord = (
                ((history == image_id) & obstacles)
                .nonzero(as_tuple=False)
                .median(dim=0)
                .values.int()
            )
        else:
            print("others")
            image_coord = (history == image_id).nonzero(as_tuple=False).median(dim=0).values.int()
        xy = self.grid_coords_to_xy(image_coord)
        return torch.Tensor([xy[0], xy[1], 1])

    def get_frontier_ids(self, xyt, planner):
        """
        This function figures out which of images correspond to an unexplored frontier.
        """
        (
            history,
            _,
            _,
        ) = self.get_active_image_descriptions()
        outside_frontier = self.get_outside_frontier(xyt, planner)
        _, explored = self.get_2d_map()
        unexplored_frontier = outside_frontier & ~explored
        history = np.ma.masked_array(history, ~unexplored_frontier)
        return np.unique(history)

    def list_objects_in_an_image(
        self, image: Union[torch.Tensor, Image.Image, np.ndarray], max_tries: int = 3
    ):
        """
        Extract visual clues (a list of featured objects) from the image observation and add the clues to a list
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)

        prompt = "List representative objects in the image (excluding floor and wall) Limit your answer in 10 words. E.G.: a table,chairs,doors"
        messages = [pil_image, prompt]

        # self.obs_count inherited from voxel_dynamem
        objects = []
        for _ in range(max_tries):
            try:
                object_names = self.image_description_client(messages)
                objects = object_names.split(",")[:5]
            except:
                objects = []
                continue
            else:
                break

        obs_ids = self.voxel_pcd._obs_counts
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        grid_coord = list(
            self.xy_to_grid_coords(
                torch.mean(xyz[obs_ids == obs_ids.max()], dim=0)[:2].int().cpu().numpy()
            )
        )
        for i in range(len(grid_coord)):
            grid_coord[i] = int(grid_coord[i])

        if len(objects) == 0:
            self.image_descriptions.append((["object"], grid_coord))
        else:
            self.image_descriptions.append((objects, grid_coord))

        print(objects)

    def get_outside_frontier(self, xyt, planner):
        """
        This function selects the edges of currently reachable space.
        """
        obstacles, _ = self.get_2d_map()
        if len(xyt) == 3:
            xyt = xyt[:2]
        reachable_points = planner.get_reachable_points(planner.to_pt(xyt))
        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = torch.tensor(reachable_xs)
        reachable_ys = torch.tensor(reachable_ys)

        reachable_map = torch.zeros_like(obstacles)
        reachable_map[reachable_xs, reachable_ys] = 1
        reachable_map = reachable_map.to(torch.bool)
        edges = get_edges(reachable_map)
        return edges & ~reachable_map
