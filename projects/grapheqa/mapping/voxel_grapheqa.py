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
import timeit
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from scipy.ndimage import maximum_filter
from torch import Tensor

from stretch.core.interfaces import Observations
from stretch.llms import OpenaiClient
from stretch.llms.prompts import DYNAMEM_VISUAL_GROUNDING_PROMPT
from stretch.mapping.instance import InstanceMemory
from stretch.mapping.voxel import SparseVoxelMap as SparseVoxelMapBase
from stretch.mapping.voxel.voxel import VALID_FRAMES, Frame
from stretch.motion import HelloStretchIdx
from stretch.perception.captioners import BaseCaptioner
from stretch.perception.encoders import BaseImageTextEncoder
from stretch.utils.morphology import binary_dilation, binary_erosion, get_edges
from stretch.utils.point_cloud import points_in_mesh
from stretch.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from stretch.utils.voxel import scatter3d

logger = logging.getLogger(__name__)


class SparseVoxelMap(SparseVoxelMapBase):
    def __init__(
        self,
        resolution: float = 0.01,
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
        encoder: Optional[BaseImageTextEncoder] = None,
        captioner: Optional[BaseCaptioner] = None,
        map_2d_device: str = "cpu",
        device: Optional[str] = None,
        use_instance_memory: bool = True,
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
        log="test",
        mllm=False,
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
            median_filter_max_error=median_filter_size,
            use_derivative_filter=use_derivative_filter,
            derivative_filter_threshold=derivative_filter_threshold,
            prune_detected_objects=prune_detected_objects,
            add_local_radius_every_step=add_local_radius_every_step,
            min_points_per_voxel=min_points_per_voxel,
            use_negative_obstacles=use_negative_obstacles,
        )

        self.point_update_threshold = point_update_threshold
        self._history_soft: Optional[Tensor] = None
        self.encoder = encoder
        self.captioner = captioner
        self.instances = InstanceMemory(
            num_envs=1,
            encoder=self.encoder,
            captioner=self.captioner,
            mask_cropped_instances=False,
            du_scale=1,
            save_original_image=True,
        )
        self.obs_count = 0
        self.log = log

        # Voxel map cache
        self._map2d: Optional[Any] = None
        self._2d_last_updated: Optional[Any] = -1

        self.mllm = mllm

        if self.mllm:
            self.gpt_client = OpenaiClient(
                DYNAMEM_VISUAL_GROUNDING_PROMPT, model="gpt-4o-2024-05-13"
            )

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
        self, debug: bool = False, return_history_id: bool = False
    ) -> Tuple[Tensor, ...]:
        """Get 2d map with explored area and frontiers."""

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
        history_soft = torch.from_numpy(maximum_filter(history_soft.float().numpy(), size=7))

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

    def add_obs(
        self,
        obs: Observations,
        camera_K: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """Unpack an observation and convert it properly, then add to memory. Pass all other inputs into the add() function as provided."""
        rgb = self.fix_type(obs.rgb)
        depth = self.fix_type(obs.depth)
        xyz = self.fix_type(obs.xyz)
        camera_pose = self.fix_type(obs.camera_pose)
        base_pose = torch.from_numpy(np.array([obs.gps[0], obs.gps[1], obs.compass[0]])).float()
        K = self.fix_type(obs.camera_K) if camera_K is None else camera_K
        task_obs = obs.task_observations

        # Allow task_observations to provide semantic sensor
        def _pop_with_task_obs_default(k, default=None):
            if task_obs is None:
                return None
            res = kwargs.pop(k, task_obs.get(k, None))
            if res is not None:
                res = self.fix_type(res)
            return res

        if task_obs is not None:
            instance_image = _pop_with_task_obs_default("instance_image")
            instance_classes = _pop_with_task_obs_default("instance_classes")
            instance_scores = _pop_with_task_obs_default("instance_scores")
        else:
            instance_image, instance_classes, instance_scores = None, None, None

        t0 = timeit.default_timer()

        self.add(
            camera_pose=camera_pose,
            xyz=xyz,
            rgb=rgb,
            depth=depth,
            base_pose=base_pose,
            obs=obs,
            camera_K=K,
            instance_image=instance_image,
            instance_classes=instance_classes,
            instance_scores=instance_scores,
            *args,
            **kwargs,
        )  # type: ignore

        t1 = timeit.default_timer()
        print("Image processing takes", t1 - t0, "seconds")

    def add(
        self,
        camera_pose: Tensor,
        rgb: Tensor,
        xyz: Optional[Tensor] = None,
        camera_K: Optional[Tensor] = None,
        feats: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        base_pose: Optional[Tensor] = None,
        instance_image: Optional[Tensor] = None,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        obs: Optional[Observations] = None,
        xyz_frame: str = "camera",
        pose_correction: Optional[Tensor] = None,
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
            instance_image(Tensor): [H,W] image of ints where values at a pixel correspond to instance_id
            instance_classes(Tensor): [K] tensor of ints where class = instance_classes[instance_id]
            instance_scores: [K] of detection confidence score = instance_scores[instance_id]
            # obs: observations
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        # Shape checking
        assert rgb.ndim == 3 or rgb.ndim == 2, f"{rgb.ndim=}: must be 2 or 3"
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        if isinstance(camera_pose, np.ndarray):
            camera_pose = torch.from_numpy(camera_pose)
        if self.use_instance_memory:
            assert rgb.ndim == 3, f"{rgb.ndim=}: must be 3 if using instance memory"
            H, W, _ = rgb.shape
            if instance_image is None:
                assert (
                    obs is not None
                ), "must provide instance image or raw observations with instances"
                assert (
                    obs.instance is not None
                ), "must provide instance image in observation if not available otherwise"
                if isinstance(obs.instance, np.ndarray):
                    instance_image = torch.from_numpy(obs.instance)
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
            median_depth = torch.from_numpy(
                scipy.ndimage.median_filter(depth, size=self.median_filter_size)
            )
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

        if pose_correction is not None:
            full_world_xyz = full_world_xyz @ pose_correction[:3, :3].T + pose_correction[:3, 3]

        # add observations before we start changing things
        self.observations.append(
            Frame(
                camera_pose,
                camera_K,
                xyz,
                rgb,
                feats,
                depth,
                instance_image,
                instance_classes,
                instance_scores,
                base_pose,
                info,
                obs,
                full_world_xyz,
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

        # Add instance views to memory
        if self.use_instance_memory:
            # Add to instance memory
            t0 = timeit.default_timer()
            instance = instance_image.clone()

            self.instances.process_instances_for_env(
                env_id=0,
                instance_seg=instance,
                point_cloud=full_world_xyz.reshape(H, W, 3),
                image=rgb.permute(2, 0, 1),
                cam_to_world=camera_pose,
                instance_classes=instance_classes,
                instance_scores=instance_scores,
                background_instance_labels=[self.background_instance_label],
                valid_points=valid_depth,
                pose=base_pose,
            )
            t1 = timeit.default_timer()
            self.instances.associate_instances_to_memory()
            if self.debug_instance_memory_processing_time:
                t2 = timeit.default_timer()
                print(__file__, ": Instance memory processing time: ", t1 - t0, t2 - t1)

        if self.prune_detected_objects:
            valid_depth = valid_depth & (instance_image == self.background_instance_label)

        # Add to voxel grid
        if feats is not None:
            feats = feats[valid_depth].reshape(-1, feats.shape[-1])
        rgb = rgb[valid_depth].reshape(-1, 3)
        world_xyz = full_world_xyz.view(-1, 3)[valid_depth.flatten()]

        # TODO: weights could also be confidence, inv distance from camera, etc
        if world_xyz.nelement() > 0:
            # Remove points that are too close to the robot
            if obs is not None and obs.joint is not None:
                state = obs.joint
                cfg = {}
                for k in HelloStretchIdx.name_to_idx:
                    cfg[k] = state[HelloStretchIdx.name_to_idx[k]]
                lk_cfg = {
                    "joint_wrist_yaw": cfg["wrist_yaw"],
                    "joint_wrist_pitch": cfg["wrist_pitch"],
                    "joint_wrist_roll": cfg["wrist_roll"],
                    "joint_lift": cfg["lift"],
                    "joint_arm_l0": cfg["arm"] / 4,
                    "joint_arm_l1": cfg["arm"] / 4,
                    "joint_arm_l2": cfg["arm"] / 4,
                    "joint_arm_l3": cfg["arm"] / 4,
                    "joint_head_pan": cfg["head_pan"],
                    "joint_head_tilt": cfg["head_tilt"],
                }
                if "gripper" in cfg.keys():
                    lk_cfg["joint_gripper_finger_left"] = cfg["gripper"]
                    lk_cfg["joint_gripper_finger_right"] = cfg["gripper"]

                mesh = self.urdf_visualizer.get_combined_robot_mesh(cfg=lk_cfg, use_collision=True)

                selected_indices = points_in_mesh(world_xyz, mesh, base_pose)
            else:
                selected_indices = torch.ones_like(world_xyz[:, 0], dtype=torch.bool)
            world_xyz = world_xyz[selected_indices]

            if feats is not None:
                feats = feats[selected_indices]

            rgb = rgb[selected_indices]

            self.voxel_pcd.add(
                world_xyz,
                features=feats,
                rgb=rgb,
                weights=None,
                min_weight_per_voxel=self._min_points_per_voxel,
            )

        if self._add_local_radius_points and (
            len(self.observations) < 2 or self._add_local_radius_every_step
        ):
            # Only do this at the first step, never after it.
            # TODO: just get this from camera_pose?
            # Add local radius points to the map around base
            if base_pose is not None:
                self._update_visited(base_pose.to(self.map_2d_device))
            else:
                # Camera only
                self._update_visited(camera_pose[:3, 3].to(self.map_2d_device))

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

    def write_to_pickle(self, filename: Optional[str] = None) -> None:
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts.

        Args:
        name (Optional[str], optional): Filename to write to. Defaults to None.
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
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("write all data to", filename)
