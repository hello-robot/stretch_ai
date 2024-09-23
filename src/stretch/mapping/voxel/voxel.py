# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import pickle
import timeit
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as open3d
import scipy
import skimage
import torch
import tqdm
from torch import Tensor

import stretch.utils.compression as compression
import stretch.utils.logger as logger
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.mapping.grid import GridParams
from stretch.mapping.instance import Instance, InstanceMemory
from stretch.motion import Footprint, PlanResult, RobotModel
from stretch.perception.encoders import BaseImageTextEncoder
from stretch.perception.wrapper import OvmmPerception
from stretch.utils.data_tools.dict import update
from stretch.utils.morphology import binary_dilation, binary_erosion, get_edges
from stretch.utils.point_cloud import create_visualization_geometries, numpy_to_pcd
from stretch.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from stretch.utils.visualization import create_disk
from stretch.utils.voxel import VoxelizedPointcloud, scatter3d

Frame = namedtuple(
    "Frame",
    [
        "camera_pose",
        "camera_K",
        "xyz",
        "rgb",
        "feats",
        "depth",
        "instance",
        "instance_classes",
        "instance_scores",
        "base_pose",
        "info",
        "obs",
        "full_world_xyz",
        "xyz_frame",
    ],
)

VALID_FRAMES = ["camera", "world"]


def ensure_tensor(arr):
    if isinstance(arr, np.ndarray):
        return Tensor(arr)
    if not isinstance(arr, Tensor):
        raise ValueError(f"arr of unknown type ({type(arr)}) cannot be cast to Tensor")


class SparseVoxelMap(object):
    """Create a voxel map object which captures 3d information.

    This class represents a 3D voxel map used for capturing environmental information. It provides various parameters
    for configuring the map's properties, such as resolution, feature dimensions, and instance memory settings.

    Attributes:
        resolution (float): The size of a voxel in meters.
        feature_dim (int): The size of feature embeddings to capture per-voxel point, separate from instance memory.
        grid_size (Tuple[int, int]): The dimensions of the voxel grid (optional).
        grid_resolution (float): The resolution of the grid (optional).
        obs_min_height (float): The minimum height for observations.
        obs_max_height (float): The maximum height for observations.
        obs_min_density (float): The minimum density for observations.
        smooth_kernel_size (int): The size of the smoothing kernel.
        add_local_radius_points (bool): Whether to add local radius points.
        remove_visited_from_obstacles(bool): subtract out observations potentially of the robot
        local_radius (float): The radius for local points.
        min_depth (float): The minimum depth for observations.
        max_depth (float): The maximum depth for observations.
        pad_obstacles (int): Padding for obstacles.
        background_instance_label (int): The label for the background instance.
        instance_memory_kwargs (Dict[str, Any]): Additional instance memory configuration.
        voxel_kwargs (Dict[str, Any]): Additional voxel configuration.
        encoder (Optional[BaseImageTextEncoder]): An encoder for feature embeddings (optional).
        map_2d_device (str): The device for 2D mapping.
        use_instance_memory (bool): Whether to create object-centric instance memory.
        min_points_per_voxel (int): The minimum number of points per voxel.
    """

    DEFAULT_INSTANCE_MAP_KWARGS = dict(
        du_scale=1,
        instance_association="bbox_iou",
        log_dir_overwrite_ok=True,
        mask_cropped_instances="False",
    )
    debug_valid_depth: bool = False
    debug_instance_memory_processing_time: bool = False
    use_negative_obstacles: bool = False

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
        local_radius: float = 0.15,
        min_depth: float = 0.1,
        max_depth: float = 4.0,
        pad_obstacles: int = 0,
        background_instance_label: int = -1,
        instance_memory_kwargs: Dict[str, Any] = {},
        voxel_kwargs: Dict[str, Any] = {},
        encoder: Optional[BaseImageTextEncoder] = None,
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
    ):
        """
        Args:
            resolution(float): in meters, size of a voxel
            feature_dim(int): size of feature embeddings to capture per-voxel point (separate from instance memory)
            use_instance_memory(bool): if we should create object-centric instance memory
            grid_size(Tuple[int, int]): dimensions of the voxel grid (optional)
            grid_resolution(float): resolution of the grid in meters, e.g. 0.05 = 5cm (optional)
            obs_min_height(float): minimum height for observations in meters
            obs_max_height(float): maximum height for observations in meters
            obs_min_density(float): minimum density for observations
            smooth_kernel_size(int): size of the smoothing kernel (in grid cells)
            add_local_radius_points(bool): whether to add local radius points to the explored map, marking them as safe
            remove_visited_from_obstacles(bool): subtract out observations potentially of the robot
            local_radius(float): radius for local points in meters
            min_depth(float): minimum depth for observations in meters
            max_depth(float): maximum depth for observations in meters
            pad_obstacles(int): padding for obstacles in grid cells
            background_instance_label(int): label for the background instance (e.g. -1)
            instance_memory_kwargs(Dict[str, Any]): additional instance memory configuration
            voxel_kwargs(Dict[str, Any]): additional voxel configuration
            encoder(BaseImageTextEncoder): encoder for feature embeddings, maps image and text to feature space (optional)
            map_2d_device(str): device for 2D mapping
            use_median_filter(bool): whether to use a median filter to remove bad depth values when mapping and exploring
            median_filter_size(int): size of the median filter
            median_filter_max_error(float): maximum error for the median filter
            use_derivative_filter(bool): whether to use a derivative filter to remove bad depth values when mapping and exploring
            derivative_filter_threshold(float): threshold for the derivative filter
            prune_detected_objects(bool): whether to prune detected objects from the voxel map
        """
        # TODO: We an use fastai.store_attr() to get rid of this boilerplate code
        self.feature_dim = feature_dim
        self.obs_min_height = obs_min_height
        self.obs_max_height = obs_max_height
        self.neg_obs_height = neg_obs_height
        self.obs_min_density = obs_min_density
        self.prune_detected_objects = prune_detected_objects

        # Smoothing kernel params
        self.smooth_kernel_size = smooth_kernel_size
        if self.smooth_kernel_size > 0:
            self.smooth_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(smooth_kernel_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.smooth_kernel = None

        # Median filter params
        self.median_filter_size = median_filter_size
        self.use_median_filter = use_median_filter
        self.median_filter_max_error = median_filter_max_error

        # Derivative filter params
        self.use_derivative_filter = use_derivative_filter
        self.derivative_filter_threshold = derivative_filter_threshold

        self.grid_resolution = grid_resolution
        self.voxel_resolution = resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pad_obstacles = int(pad_obstacles)
        self.background_instance_label = background_instance_label
        self.instance_memory_kwargs = update(
            copy.deepcopy(self.DEFAULT_INSTANCE_MAP_KWARGS), instance_memory_kwargs
        )
        self.use_instance_memory = use_instance_memory
        self.voxel_kwargs = voxel_kwargs
        self.encoder = encoder
        self.map_2d_device = map_2d_device
        self._min_points_per_voxel = min_points_per_voxel

        # Set the device we use for things here
        if device is not None:
            self.device = device
        else:
            self.device = self.map_2d_device

        # Create kernel(s) for obstacle dilation over 2d/3d maps
        if self.pad_obstacles > 0:
            self.dilate_obstacles_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(self.pad_obstacles))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_obstacles_kernel = None

        # Add points with local_radius to the voxel map at (0,0,0) unless we receive lidar points
        self._add_local_radius_points = add_local_radius_points
        self._add_local_radius_every_step = add_local_radius_every_step
        self._remove_visited_from_obstacles = remove_visited_from_obstacles
        self.local_radius = local_radius

        # Create disk for mapping explored areas near the robot - since camera can't always see it
        self._disk_size = np.ceil(self.local_radius / self.grid_resolution)

        self._visited_disk = torch.from_numpy(
            create_disk(self._disk_size, (2 * self._disk_size) + 1)
        ).to(map_2d_device)

        self.grid = GridParams(grid_size=grid_size, resolution=resolution, device=map_2d_device)
        self.grid_size = self.grid.grid_size
        self.grid_origin = self.grid.grid_origin
        self.resolution = self.grid.resolution

        # Init variables
        self.reset()

    def reset(self) -> None:
        """Clear out the entire voxel map."""
        self.observations = []
        # Create an instance memory to associate bounding boxes in space
        if self.use_instance_memory:
            self.instances = InstanceMemory(
                num_envs=1,
                encoder=self.encoder,
                **self.instance_memory_kwargs,
            )
        else:
            self.instances = None

        # Create voxelized pointcloud
        self.voxel_pcd = VoxelizedPointcloud(
            voxel_size=self.voxel_resolution,
            dim_mins=None,
            dim_maxs=None,
            feature_pool_method="mean",
            **self.voxel_kwargs,
        )

        self._seq = 0
        self._2d_last_updated = -1
        # Create map here - just reset *some* variables
        self.reset_cache()

    def reset_cache(self):
        """Clear some tracked things"""
        # Stores points in 2d coords where robot has been
        self._visited = torch.zeros(self.grid_size, device=self.map_2d_device)

        # Store instances detected (all of them for now)
        if self.use_instance_memory:
            self.instances.reset()

        self.voxel_pcd.reset()

        # Store 2d map information
        # This is computed from our various point clouds
        self._map2d = None

    def get_instances(self) -> List[Instance]:
        """Return a list of all viewable instances"""
        if self.instances is None:
            return []
        return list(self.instances.instances[0].values())

    def get_instances_as_dict(self) -> Dict[int, Instance]:
        """Return a dictionary of all viewable instances"""
        if self.instances is None:
            return {}
        return self.instances.instances[0]

    def fix_type(self, tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert to tensor and float"""
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        return tensor.float()

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
        )

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
            # trimesh.transform_points(xyz, camera_pose)
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

            if self.debug_valid_depth:
                # This is a block of debug code for displaying valid depths, in case for some reason valid regions and objects are being rejected out of hand for no good reason.
                print("valid_depth", valid_depth.sum(), valid_depth.shape)
                import matplotlib

                matplotlib.use("TkAgg")
                import matplotlib.pyplot as plt

                plt.subplot(121)
                plt.imshow(valid_depth.cpu().numpy())
                valid_depth_mask = valid_depth[:, :, None].repeat([1, 1, 3])
                plt.subplot(122)
                plt.imshow(valid_depth_mask.cpu().numpy() * rgb.cpu().numpy().astype(np.uint8))
                plt.show()

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

    def mask_from_bounds(self, bounds: np.ndarray, debug: bool = False):
        """create mask from a set of 3d object bounds"""
        assert bounds.shape[0] == 3, "bounding boxes in xyz"
        assert bounds.shape[1] == 2, "min and max"
        assert (len(bounds.shape)) == 2, "only one bounding box"
        obstacles, explored = self.get_2d_map()
        return self.grid.mask_from_bounds(obstacles, explored, bounds, debug)

    def _update_visited(self, base_pose: Tensor):
        """Update 2d map of where robot has visited"""
        # Add exploration here
        # Base pose can be whatever, going to assume xyt for now
        map_xy = ((base_pose[:2] / self.grid_resolution) + self.grid_origin[:2]).int()
        x0 = int(map_xy[0] - self._disk_size)
        x1 = int(map_xy[0] + self._disk_size + 1)
        y0 = int(map_xy[1] - self._disk_size)
        y1 = int(map_xy[1] + self._disk_size + 1)
        assert x0 >= 0
        assert y0 >= 0
        self._visited[x0:x1, y0:y1] += self._visited_disk

    def write_to_pickle(self, filename: str, compress: bool = True) -> None:
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
        data = {}
        data["camera_poses"] = []
        data["camera_K"] = []
        data["base_poses"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        data["instance"] = []
        data["instance_scores"] = []
        data["instance_classes"] = []

        # Add a print statement with use of this code
        logger.alert(f"Write pkl to {filename}...")
        logger.alert(f"You may visualize this file with:")
        logger.alert()
        logger.alert(f"\tpython -m stretch.app.read_map -i {filename} --show-svm")
        logger.alert()

        for frame in tqdm.tqdm(
            self.observations, desc="Aggregating data to write", unit="frame", ncols=80
        ):
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["camera_K"].append(frame.camera_K)
            data["instance"].append(frame.instance)
            data["instance_classes"].append(frame.instance_classes)
            data["instance_scores"].append(frame.instance_scores)

            # Convert to numpy and correct formats for saving
            rgb = frame.rgb.byte().cpu().numpy()
            depth = (frame.depth * 1000).cpu().numpy().astype(np.uint16)

            # Handle compression
            if compress:
                data["rgb"].append(compression.to_jpg(rgb))
                data["depth"].append(compression.to_jp2(depth))
            else:
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
        ) = self.voxel_pcd.get_pointcloud()
        data["compressed"] = compress
        print("Dumping to pickle...")
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def fix_data_type(self, tensor) -> torch.Tensor:
        """make sure tensors are in the right format for this model"""
        # If its empty just hope we're handling that somewhere else
        if tensor is None:
            return None
        # Conversions
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        # Data types
        if isinstance(tensor, torch.Tensor):
            return tensor.float()
        else:
            raise NotImplementedError("unsupported data type for tensor:", tensor)

    def read_from_pickle(
        self,
        filename: str,
        num_frames: int = -1,
        perception: Optional[OvmmPerception] = None,
        transform_pose: Optional[torch.Tensor] = None,
        reset: bool = False,
    ) -> bool:
        """Read from a pickle file as above. Will clear all currently stored data first.

        Args:
            filename(str): path to the pickle file
            num_frames(int): number of frames to read from the file
            perception(OvmmPerception): perception model to use for instance segmentation
            transform_pose(torch.Tensor): transformation to apply to camera poses
            reset(bool): whether to clear all currently stored data first
        """
        if reset:
            self.reset_cache()
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.exists(), f"No file found at {filename}"
        with filename.open("rb") as f:
            data = pickle.load(f)

        # Flag for if the data is compressed
        compressed = False
        if "compressed" in data:
            compressed = data["compressed"]
        read_observations = False
        if "obs" in data:
            logger.warning("Reading old format with full observations")
            read_observations = True

        # Processing to handle older files that actually saved the whole observation object
        if read_observations and len(data["obs"]) > 0:
            instance_data = data["obs"]
        else:
            instance_data = data["instance"]

        if len(instance_data) == 0:
            logger.error("No instance data found in file")
            return False

        for i, (camera_pose, K, rgb, feats, depth, base_pose, instance) in enumerate(
            tqdm.tqdm(
                # TODO: compression of observations
                # Right now we just do not support this
                # data["obs"],  TODO: compression of Observations
                zip(
                    data["camera_poses"],
                    data["camera_K"],
                    data["rgb"],
                    data["feats"],
                    data["depth"],
                    data["base_poses"],
                    instance_data,
                ),
                ncols=80,
                desc="Reading data from pickle",
                unit="frame",
            )
        ):
            # Handle the case where we dont actually want to load everything
            if num_frames > 0 and i >= num_frames:
                break
            if camera_pose is None:
                logger.warning(f"Skipping frame {i} with None camera pose")
                continue
            if K is None:
                logger.warning(f"Skipping frame {i} with None intrinsics")
                continue
            if base_pose is None:
                logger.warning(f"Skipping frame {i} with None base pose")
                continue

            camera_pose = self.fix_data_type(camera_pose)
            if compressed:
                rgb = compression.from_jpg(rgb)
                depth = compression.from_jp2(depth) / 1000.0
            rgb = self.fix_data_type(rgb).float()
            depth = self.fix_data_type(depth).float()
            if feats is not None:
                feats = self.fix_data_type(feats)

            base_pose = self.fix_data_type(base_pose)

            # Handle instance processing - if we have a perception model we can use it to predict the instance image
            # We can also just use the instance image if it was saved
            instance_classes = None
            instance_scores = None
            if perception is not None:
                _, instance, info = perception.predict_segmentation(
                    rgb=rgb, depth=depth, base_pose=base_pose
                )
                instance_classes = info["instance_classes"]
                instance_scores = info["instance_scores"]
            elif read_observations:
                instance = instance.instance
                instance_classes = instance.task_observations["instance_classes"]
                instance_scores = instance.task_observations["instance_scores"]
            else:
                instance_classes = self.fix_data_type(data["instance_classes"][i])
                instance_scores = self.fix_data_type(data["instance_scores"][i])
            if instance is not None:
                instance = self.fix_data_type(instance).long()

            # Add to the map
            self.add(
                camera_pose=camera_pose,
                rgb=rgb,
                feats=feats,
                depth=depth,
                base_pose=base_pose,
                instance_image=instance,
                instance_classes=instance_classes,
                instance_scores=instance_scores,
                camera_K=K,
                pose_correction=transform_pose,
            )
        return True

    def recompute_map(self):
        """Recompute the entire map from scratch instead of doing incremental updates.
        This is a helper function which recomputes everything from the beginning.

        Currently this will be slightly inefficient since it recreates all the objects incrementally.
        """
        old_observations = self.observations
        self.reset_cache()
        for frame in old_observations:
            self.add(
                frame.camera_pose,
                frame.xyz,
                frame.rgb,
                frame.feats,
                frame.depth,
                frame.base_pose,
                frame.obs,
                **frame.info,
            )

    def get_pointcloud(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the current point cloud"""
        return self.voxel_pcd.get_pointcloud()

    def get_2d_map(self, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2d map with explored area and frontiers."""

        # Is this already cached? If so we don't need to go to all this work
        if self._map2d is not None and self._seq == self._2d_last_updated:
            return self._map2d

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz, _, counts, _ = self.voxel_pcd.get_pointcloud()

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin).long()

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        grid_size = self.grid_size + [max_height]
        voxels = torch.zeros(grid_size, device=device)

        # Mask out obstacles only above a certain height
        obs_mask = xyz[:, -1] < max_height
        if self.use_negative_obstacles:
            neg_height = int(self.neg_obs_height / self.grid_resolution)
            negative_obstacles = xyz[:, -1] < neg_height
            obs_mask = obs_mask | negative_obstacles
        xyz = xyz[obs_mask, :]
        counts = counts[obs_mask][:, None]

        # voxels[x_coords, y_coords, z_coords] = 1
        voxels = scatter3d(xyz, counts, grid_size).squeeze()

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:]
        obstacles_soft = torch.sum(obstacle_voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density

        if self._remove_visited_from_obstacles:
            # Remove "visited" points containing observations of the robot
            obstacles *= (1 - self._visited).bool()

        if self.dilate_obstacles_kernel is not None:
            obstacles = binary_dilation(
                obstacles.float().unsqueeze(0).unsqueeze(0),
                self.dilate_obstacles_kernel,
            )[0, 0].bool()

        # Explored area = only floor mass
        explored_soft = torch.sum(voxels, dim=-1)

        # Add explored radius around the robot, up to min depth
        # TODO: make sure lidar is supported here as well; if we do not have lidar assume a certain radius is explored
        explored_soft += self._visited
        explored = explored_soft > 0

        if self.smooth_kernel_size > 0:
            # Opening and closing operations here on explore
            explored = binary_erosion(
                binary_dilation(explored.float().unsqueeze(0).unsqueeze(0), self.smooth_kernel),
                self.smooth_kernel,
            )  # [0, 0].bool()
            explored = binary_dilation(
                binary_erosion(explored, self.smooth_kernel),
                self.smooth_kernel,
            )[0, 0].bool()

            # Obstacles just get dilated and eroded
            obstacles = binary_erosion(
                binary_dilation(obstacles.float().unsqueeze(0).unsqueeze(0), self.smooth_kernel),
                self.smooth_kernel,
            )[0, 0].bool()

        if debug:
            import matplotlib.pyplot as plt

            # TODO: uncomment to show the original world representation
            # from stretch.utils.point_cloud import show_point_cloud
            # show_point_cloud(xyz, rgb / 255., orig=np.zeros(3))
            # TODO: uncomment to show voxel point cloud
            # from stretch.utils.point_cloud import show_point_cloud
            # show_point_cloud(xyz, rgb/255., orig=self.grid_origin)

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

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        return obstacles, explored

    def plan_to_grid_coords(self, plan_result: PlanResult) -> Optional[List[torch.Tensor]]:
        """Convert a plan properly into grid coordinates"""
        if not plan_result.success:
            return None
        else:
            traj = []
            for node in plan_result.trajectory:
                traj.append(self.grid.xy_to_grid_coords(node.state[:2]))
            return traj

    def get_kd_tree(self) -> open3d.geometry.KDTreeFlann:
        """Return kdtree for collision checks

        We could use Kaolin to get octree from pointcloud.
        Not hard to parallelize on GPU:
            Octree has K levels, each cube in level k corresponds to a  regular grid of "supervoxels"
            Occupancy can be done for each level in parallel.
        Hard part is converting to KDTreeFlann (or modifying the collision check to run on gpu)
        """
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        pcd = numpy_to_pcd(points.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        return open3d.geometry.KDTreeFlann(pcd)

    def show(self, instances: bool = False, **backend_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        if instances:
            assert self.use_instance_memory, "must have instance memory to show instances"
        return self._show_open3d(instances, **backend_kwargs)

    def get_xyz_rgb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return xyz and rgb of the current map"""
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        return points, rgb

    def sample_explored(self) -> Optional[np.ndarray]:
        """Return obstacle-free xy point in explored space"""
        obstacles, explored = self.get_2d_map()
        return self.sample_from_mask(~obstacles & explored)

    def sample_from_mask(self, mask: torch.Tensor) -> Optional[np.ndarray]:
        """Sample from any mask"""
        valid_indices = torch.nonzero(mask, as_tuple=False)
        if valid_indices.size(0) > 0:
            random_index = torch.randint(valid_indices.size(0), (1,))
            return self.grid.grid_coords_to_xy(valid_indices[random_index])
        else:
            return None

    def xyt_is_safe(self, xyt: np.ndarray, robot: Optional[RobotModel] = None) -> bool:
        """Check to see if a given xyt position is known to be safe."""
        if robot is not None:
            raise NotImplementedError("not currently checking against robot base geometry")
        obstacles, explored = self.get_2d_map()
        # Convert xy to grid coords
        grid_xy = self.grid.xy_to_grid_coords(xyt[:2])
        # Check to see if grid coords are explored and obstacle free
        if grid_xy is None:
            # Conversion failed - probably out of bounds
            return False
        obstacles, explored = self.get_2d_map()
        # Convert xy to grid coords
        grid_xy = self.grid.xy_to_grid_coords(xyt[:2])
        # Check to see if grid coords are explored and obstacle free
        if grid_xy is None:
            # Conversion failed - probably out of bounds
            return False
        if robot is not None:
            # TODO: check against robot geometry
            raise NotImplementedError("not currently checking against robot base geometry")
        return True

    def postprocess_instances(self):
        self.instances.global_box_compression_and_nms(env_id=0)

    def _get_boxes_from_points(
        self,
        traversible: torch.Tensor,
        color: List[float],
        is_map: bool = True,
        height: float = 0.0,
        offset: Optional[np.ndarray] = None,
    ):
        """Get colored boxes for a mask"""
        # Get indices for all traversible locations
        traversible_indices = np.argwhere(traversible)
        # Traversible indices will be a 2xN array, so we need to transpose it.
        # Set to floor/max obs height and bright red
        if is_map:
            traversible_pts = self.grid.grid_coords_to_xy(traversible_indices.T)
        else:
            traversible_pts = (
                traversible_indices.T - np.ceil([d / 2 for d in traversible.shape])
            ) * self.grid_resolution
        if offset is not None:
            traversible_pts += offset

        geoms = []
        for i in range(traversible_pts.shape[0]):
            center = np.array(
                [
                    traversible_pts[i, 0],
                    traversible_pts[i, 1],
                    self.obs_min_height + height,
                ]
            )
            dimensions = np.array(
                [self.grid_resolution, self.grid_resolution, self.grid_resolution]
            )

            # Create a custom geometry with red color
            mesh_box = open3d.geometry.TriangleMesh.create_box(
                width=dimensions[0], height=dimensions[1], depth=dimensions[2]
            )
            mesh_box.paint_uniform_color(color)  # Set color to red
            mesh_box.translate(center)

            # Visualize the red box
            geoms.append(mesh_box)
        return geoms

    def _get_open3d_geometries(
        self,
        instances: bool,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        xyt: Optional[np.ndarray] = None,
        footprint: Optional[Footprint] = None,
        add_planner_visuals: bool = True,
        **backend_kwargs,
    ):
        """Show and return bounding box information and rgb color information from an explored point cloud. Uses open3d."""

        # Create a combined point cloud
        # Do the other stuff we need to show instances
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        pcd = numpy_to_pcd(points.detach().cpu().numpy(), (rgb / norm).detach().cpu().numpy())
        if orig is None:
            orig = np.zeros(3)
        geoms = create_visualization_geometries(pcd=pcd, orig=orig)

        # Get the explored/traversible area
        obstacles, explored = self.get_2d_map()
        traversible = explored & ~obstacles

        if add_planner_visuals:
            geoms += self._get_boxes_from_points(traversible, [0, 1, 0])
            geoms += self._get_boxes_from_points(obstacles, [1, 0, 0])

            if xyt is not None and footprint is not None:
                geoms += self._get_boxes_from_points(
                    footprint.get_rotated_mask(self.grid_resolution, float(xyt[2])),
                    [0, 0, 1],
                    is_map=False,
                    height=0.1,
                    offset=xyt[:2],
                )

        if instances:
            self._get_instances_open3d(geoms)

        return geoms

    def _get_instances_open3d(self, geoms: List[open3d.geometry.Geometry]) -> None:
        """Get open3d geometries to append"""
        for instance_view in self.get_instances():
            mins, maxs = (
                instance_view.bounds[:, 0].cpu().numpy(),
                instance_view.bounds[:, 1].cpu().numpy(),
            )
            if np.any(maxs - mins < 1e-5):
                logger.info(f"Warning: bad box: {mins} {maxs}")
                continue
            width, height, depth = maxs - mins

            # Create a mesh to visualzie where the instances were seen
            mesh_box = open3d.geometry.TriangleMesh.create_box(
                width=width, height=height, depth=depth
            )

            # Get vertex array from the mesh
            vertices = np.asarray(mesh_box.vertices)

            # Translate the vertices to the desired position
            vertices += mins
            triangles = np.asarray(mesh_box.triangles)

            # Create a wireframe mesh
            lines = []
            for tri in triangles:
                lines.append([tri[0], tri[1]])
                lines.append([tri[1], tri[2]])
                lines.append([tri[2], tri[0]])

            # color = [1.0, 0.0, 0.0]  # Red color (R, G, B)
            color = np.random.random(3)
            colors = [color for _ in range(len(lines))]
            wireframe = open3d.geometry.LineSet(
                points=open3d.utility.Vector3dVector(vertices),
                lines=open3d.utility.Vector2iVector(lines),
            )
            # Get the colors and add to wireframe
            wireframe.colors = open3d.utility.Vector3dVector(colors)
            geoms.append(wireframe)

    def delete_instance(self, instance: Instance) -> None:
        """Remove an instance from the map"""
        print("Deleting instance", instance.global_id)
        print("Bounds: ", instance.bounds)
        self.delete_obstacles(instance.bounds)
        self.instances.pop_global_instance(env_id=0, global_instance_id=instance.global_id)

    def delete_obstacles(
        self,
        bounds: Optional[np.ndarray] = None,
        point: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        min_height: Optional[float] = None,
    ) -> None:
        """Delete obstacles from the map"""
        self.voxel_pcd.remove(bounds, point, radius, min_height=min_height)

        # Force recompute of 2d map
        self.get_2d_map()

    def _show_open3d(
        self,
        instances: bool,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        xyt: Optional[np.ndarray] = None,
        footprint: Optional[Footprint] = None,
        planner_visuals: bool = True,
        **backend_kwargs,
    ):
        """Show and return bounding box information and rgb color information from an explored point cloud. Uses open3d."""

        # get geometries so we can use them
        geoms = self._get_open3d_geometries(
            instances, orig, norm, xyt=xyt, footprint=footprint, add_planner_visuals=planner_visuals
        )
        # Show the geometries of where we have explored
        open3d.visualization.draw_geometries(geoms)

        # Returns xyz and rgb for further inspection
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        return points, rgb

    @staticmethod
    def from_parameters(
        parameters: Parameters,
        encoder: BaseImageTextEncoder,
        voxel_size: float = 0.05,
        use_instance_memory: bool = True,
        **kwargs,
    ) -> "SparseVoxelMap":
        return SparseVoxelMap(
            resolution=voxel_size,
            local_radius=parameters["local_radius"],
            grid_resolution=parameters["voxel_size"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            neg_obs_height=parameters["neg_obs_height"],
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            add_local_radius_every_step=parameters["add_local_every_step"],
            min_points_per_voxel=parameters["min_points_per_voxel"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            obs_min_density=parameters["obs_min_density"],
            encoder=encoder,
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            use_instance_memory=use_instance_memory,
            instance_memory_kwargs={
                "min_pixels_for_instance_view": parameters.get("min_pixels_for_instance_view", 100),
                "min_instance_thickness": parameters.get(
                    "instance_memory/min_instance_thickness", 0.01
                ),
                "min_instance_vol": parameters.get("instance_memory/min_instance_vol", 1e-6),
                "max_instance_vol": parameters.get("instance_memory/max_instance_vol", 10.0),
                "min_instance_height": parameters.get("instance_memory/min_instance_height", 0.1),
                "max_instance_height": parameters.get("instance_memory/max_instance_height", 1.8),
                "min_pixels_for_instance_view": parameters.get(
                    "instance_memory/min_pixels_for_instance_view", 100
                ),
                "min_percent_for_instance_view": parameters.get(
                    "instance_memory/min_percent_for_instance_view", 0.2
                ),
                "mask_cropped_instances": parameters.get(
                    "instance_memory/mask_cropped_instances", False
                ),
                "open_vocab_cat_map_file": parameters.get("open_vocab_category_map_file", None),
                "use_visual_feat": parameters.get("use_visual_feat", False),
            },
            prune_detected_objects=parameters.get("prune_detected_objects", False),
        )

    def _get_instance_color(instance_id: int) -> List[float]:
        """Get a color for an instance"""
        return [np.random.random() for _ in range(3)]
