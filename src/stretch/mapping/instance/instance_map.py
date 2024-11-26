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

import os
import shutil
import timeit
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from stretch.mapping.instance import Instance, InstanceView
from stretch.mapping.instance.matching import ViewMatchingConfig, get_similarity
from stretch.perception.encoders import BaseImageTextEncoder
from stretch.utils.bboxes_3d import (
    box3d_intersection_from_bounds,
    box3d_nms,
    box3d_volume_from_bounds,
    get_box_verts_from_bounds,
)
from stretch.utils.image import dilate_or_erode_mask, interpolate_image
from stretch.utils.logger import Logger
from stretch.utils.point_cloud_torch import get_bounds
from stretch.utils.voxel import drop_smallest_weight_points

logger = Logger(__name__)
logger.hide_info()


class InstanceMemory:
    """
    InstanceMemory stores information about instances found in the environment. It stores a list of Instance objects, each of which is a single instance found in the environment.

    images: list of egocentric images at each timestep
    instance_views: list of InstanceView objects at each timestep
    point_cloud: list of point clouds at each timestep
    unprocessed_views: list of unprocessed InstanceView objects at each timestep, before they are added to an Instance object
    timesteps: list of timesteps
    """

    images: List[Tensor] = []
    instances: List[Dict[int, Instance]] = []
    point_cloud: List[Tensor] = []
    unprocessed_views: List[Dict[int, InstanceView]] = []
    local_id_to_global_id_map: List[Dict[int, int]] = []
    timesteps: List[int] = []

    def __len__(self):
        """Figure out how many things we have found"""
        num_instances = 0
        for env in self.instances:
            num_instances += len(env)
        return num_instances

    def __init__(
        self,
        num_envs: int,
        du_scale: int,
        instance_association: str = "bbox_iou",
        instance_association_within_class: bool = True,
        iou_threshold: float = 0.8,
        overlap_eps: float = 1e-6,
        global_box_nms_thresh: float = 0.0,
        instance_box_compression_drop_prop: float = 0.1,
        instance_box_compression_resolution: float = 0.05,
        debug_visualize: bool = False,
        erode_mask_num_pix: int = 0,
        erode_mask_num_iter: int = 1,
        instance_view_score_aggregation_mode="max",
        min_pixels_for_instance_view=100,
        min_percent_for_instance_view=0.2,
        log_dir: Optional[str] = "instances",
        log_dir_overwrite_ok: bool = False,
        view_matching_config: ViewMatchingConfig = ViewMatchingConfig(),
        mask_cropped_instances: bool = True,
        crop_padding: float = 1.5,
        min_instance_vol: float = 1e-6,
        max_instance_vol: float = 10.0,
        min_instance_thickness: float = 0.01,
        min_instance_height: float = 0.1,
        max_instance_height: float = 1.8,
        min_instance_points: int = 100,
        use_visual_feat: bool = False,
        open_vocab_cat_map_file: str = None,
        encoder: Optional[BaseImageTextEncoder] = None,
    ):
        """See class definition for information about InstanceMemory

        Args:
            num_envs (int): Number of environments to track
            du_scale (int): Downsample images by 1 / du_scale
            instance_association (str, optional): Instance association method. Defaults to "bbox_iou".
            iou_threshold (float, optional): Threshold for associating instance views to global memory. Defaults to 0.8.
            global_box_nms_thresh (float): If nonzero, perform nonmax suppression on global bboxes after associating instances to memory
            debug_visualize (bool, optional): Visualize by writing out to disk. Defaults to False.
            erode_mask_num_pix (int, optional): If nonzero, how many pixels to erode instance masks. Defaults to 0.
            erode_mask_num_iter (int, optional): If erode_mask_num_pix is nonzero, how times to iterate erode instance masks. Defaults to 1.
            instance_view_score_aggregation_mode (str): When adding views to an instance, how to update instance scores. Defaults to 'max'
            mask_cropped_instances (bool): true if we want to save crops of just objects on black background; false otherwise
        """
        self.mask_cropped_instances = mask_cropped_instances
        self.num_envs = num_envs
        self.du_scale = du_scale
        self.debug_visualize = debug_visualize
        self.crop_padding = crop_padding
        # self.instance_association = instance_association
        # self.iou_threshold = iou_threshold
        self.erode_mask_num_pix = erode_mask_num_pix
        self.erode_mask_num_iter = erode_mask_num_iter
        self.global_box_nms_thresh = global_box_nms_thresh
        self.instance_box_compression_drop_prop = instance_box_compression_drop_prop
        self.instance_box_compression_resolution = instance_box_compression_resolution
        self.encoder = encoder

        self.min_instance_vol = min_instance_vol
        self.max_instance_vol = max_instance_vol
        self.min_instance_thickness = min_instance_thickness
        self.min_instance_height = min_instance_height
        self.max_instance_height = max_instance_height
        self.min_instance_points = min_instance_points
        self.use_visual_feat = use_visual_feat  # whether use visual feat to merge instances

        if isinstance(view_matching_config, dict):
            view_matching_config = ViewMatchingConfig(**view_matching_config)
        self.view_matching_config = view_matching_config

        self.instance_view_score_aggregation_mode = instance_view_score_aggregation_mode
        self.min_pixels_for_instance_view = min_pixels_for_instance_view
        self.min_percent_for_instance_view = min_percent_for_instance_view

        # Logging instance memory
        self.log_dir = log_dir
        if self.log_dir is not None:
            shutil.rmtree(self.log_dir, ignore_errors=True)
            os.makedirs(self.log_dir, exist_ok=log_dir_overwrite_ok)

        self.reset()

    def reset(self):
        """
        Reset the state of instance memory after an episode ends
        """
        self.images = [None for _ in range(self.num_envs)]
        self.point_cloud = [None for _ in range(self.num_envs)]
        self.instances = [{} for _ in range(self.num_envs)]
        self.unprocessed_views = [{} for _ in range(self.num_envs)]
        self.local_id_to_global_id_map = [{} for _ in range(self.num_envs)]
        self.timesteps = [0 for _ in range(self.num_envs)]

    def get_instance(self, global_instance_id: int, env_id: int = 0) -> Instance:
        """
        Retrieve an instance given an environment ID and a global instance ID.

        Args:
            env_id (int): The environment ID.
            global_instance_id (int): The global instance ID within the specified environment.

        Returns:
            Instance: The Instance object associated with the given IDs.
        """
        return self.instances[env_id][global_instance_id]

    def get_instances(self, env_id: int = 0) -> List[Instance]:
        """Returns a list of all global instances for a single environment"""
        global_instance_ids = self.get_global_instance_ids(env_id)
        if len(global_instance_ids) == 0:
            return []
        global_instances = self.get_instances_by_ids(
            env_id=env_id, global_instance_idxs=global_instance_ids
        )
        return global_instances

    def get_global_instance_ids(self, env_id: int) -> List[int]:
        """
        Get the list of global instance IDs associated with a given environment ID.

        Args:
            env_id (int): The environment ID.

        Returns:
            List[int]: A list of global instance IDs associated with the environment.
        """
        return list(self.instances[env_id].keys())

    def get_instances_by_ids(
        self, env_id: int, global_instance_idxs: Optional[Sequence[int]] = None
    ) -> List[Instance]:
        """
        Retrieve a list of instances for a given environment ID. If instance indexes are specified, only instances
        with those indexes will be returned.

        Args:
            env_id (int): The environment ID.
            global_instance_idxs (Optional[Sequence[int]]): The global instance IDs to retrieve. If None, all instances
                                                    for the given environment will be returned. Defaults to None.

        Returns:
            List[Instance]: List of Instance objects associated with the given IDs.
        """
        if global_instance_idxs is None:
            return list(self.instances[env_id].values())
        return [self.instances[env_id][g_id] for g_id in global_instance_idxs]

    def pop_global_instance(
        self, env_id: int, global_instance_id: int, skip_reindex: bool = False
    ) -> Instance:
        """
        Remove and return an instance given an environment ID and a global instance ID. Optionally skip reindexing of
        global instance IDs.

        Args:
            env_id (int): The environment ID.
            global_instance_id (int): The global instance ID to remove.
            skip_reindex (bool): Whether to skip reindexing of global instance IDs. Defaults to False.

        Returns:
            Instance: The removed Instance object.
        """
        instance = self.instances[env_id].pop(global_instance_id)
        if not skip_reindex:
            self.reindex_global_instances(env_id=env_id)
        return instance

    def reindex_global_instances(self, env_id: int) -> Dict[int, Instance]:
        """
        Reindex the global instance IDs within a specified environment. This is typically used after removing an instance
        to ensure that the global instance IDs are contiguous. Mutates self.instances.

        Args:
            env_id (int): The environment ID for which to reindex global instance IDs.

        Returns:
            Dict[int, Instance]: The newly indexed dictionary of Instance objects for the given environment.
        """
        if len(self.instances[env_id]) == 0:
            self.instances[env_id] = {}
            return self.instances[env_id]

        ids, instances = zip(*self.instances[env_id].items())
        new_ids = range(len(ids))
        new_env_instances = dict(zip(new_ids, instances))
        self.instances[env_id] = new_env_instances
        return self.instances[env_id]

    def get_ids_to_instances(
        self, env_id: int, category_id: Optional[int] = None
    ) -> Dict[int, Instance]:
        """
        Retrieve a Dict of IDs -> global instances for a given environment. If category_id is specified,
        only instances matching that category will be returned.

        Args:
            env_id (int): The environment ID.
            category_id: (Optional[Sequence[int]]): The category of instances to retrieve. If None, all instances
                                                    for the given environment will be returned. Defaults to None.

        Returns:
            Dict[int, Instance]: ID -> Instance objects associated with the given category_id.
        """
        # Get global instances
        global_instance_ids = self.get_global_instance_ids(env_id)
        if len(global_instance_ids) == 0:
            return {}
        global_instances = self.get_instances_by_ids(
            env_id=env_id, global_instance_idxs=global_instance_ids
        )
        return_dict = {gid: inst for gid, inst in zip(global_instance_ids, global_instances)}
        if category_id is not None:
            return_dict = {gid: g for gid, g in return_dict.items() if g.category_id == category_id}
        return return_dict

    def associate_instances_to_memory(self, debug: bool = True):
        """
        Associate instance views with existing instances or create new instances based on matching criteria.

        This method performs instance association for each instance view across environments. It determines whether an instance view
        should be added to an existing instance or a new instance should be created.

        The association process can be based on Intersection over Union (IoU) or a global map.

        For each environment and local instance view, the following steps are taken:
        - If the instance association method is set to "bbox_iou", the best matching global instance is found using the
        `find_global_instance_by_bbox_overlap` method. If a suitable global instance is not found (IoU below a threshold),
        a new instance is created. Otherwise, the instance view is associated with the existing global instance.
        - If the instance association method is set to "map_overlap", the association occurs during the global map update, and no action is taken here.
        - If the instance association method is not recognized, a NotImplementedError is raised.

        Note:
            The instance association process is critical for maintaining a coherent memory representation of instances across views.

        Raises:
            NotImplementedError: When an unrecognized instance association method is specified.
        """
        for env_id in range(self.num_envs):
            for local_instance_id, instance_view in self.unprocessed_views[env_id].items():
                match_category_id = (
                    instance_view.category_id if self.view_matching_config.within_class else None
                )
                # image_array = np.array(instance_view.cropped_image, dtype=np.uint8)
                # image_debug = Image.fromarray(image_array)
                # image_debug.show()
                global_ids_to_instances = self.get_ids_to_instances(
                    env_id, category_id=match_category_id
                )
                if len(global_ids_to_instances) == 0:
                    # Create new global instance
                    self.add_view_to_instance(
                        env_id, local_instance_id, len(self.instances[env_id])
                    )
                    continue
                global_instance_ids, global_bounds, global_embedding = zip(
                    *[
                        (
                            inst_id,
                            instance.bounds,
                            instance.get_image_embedding(
                                aggregation_method="mean",
                                use_visual_feat=self.use_visual_feat,
                                normalize=False,
                            ),
                        )  # Slow since we concatenate all global vectors each time for each image instance
                        for inst_id, instance in global_ids_to_instances.items()
                    ]
                )

                # Similarity config
                similarity = get_similarity(
                    instance_bounds1=instance_view.bounds.unsqueeze(0),
                    instance_bounds2=global_bounds,
                    visual_embedding1=instance_view.visual_feat
                    if self.use_visual_feat
                    else instance_view.embedding,
                    visual_embedding2=global_embedding,
                    text_embedding1=None,
                    text_embedding2=None,
                    view_matching_config=self.view_matching_config,
                )
                max_similarity, matched_idx = similarity.max(dim=1)
                total_weight = (
                    self.view_matching_config.visual_similarity_weight
                    + self.view_matching_config.box_overlap_weight
                )
                max_similarity = max_similarity / total_weight

                if max_similarity < self.view_matching_config.min_similarity_thresh:
                    # Create a new instance here
                    matched_global_instance_id = len(self.instances[env_id])  # + 1
                else:
                    # Do not create a new instance!
                    matched_global_instance_id = list(global_instance_ids)[matched_idx]

                self.add_view_to_instance(env_id, local_instance_id, matched_global_instance_id)

        # # TODO: Add option to do global NMS
        # if self.global_box_nms_thresh > 0.0:
        #     for env_id in range(self.num_envs):
        #         self.global_instance_nms(env_id)

    def get_local_instance_view(self, env_id: int, local_instance_id: int):
        """
        Retrieve the local instance view associated with a specific local instance in a given environment.

        This method fetches the unprocessed instance view corresponding to a specified local instance within a particular environment.

        Args:
            env_id (int): Identifier for the environment in which the instance view should be retrieved.
            local_instance_id (int): Identifier for the local instance within the specified environment.

        Returns:
            instance_view (Optional[InstanceView]): The instance view associated with the specified local instance in the given environment,
                or None if no matching instance view is found.
        """
        instance_view = self.unprocessed_views[env_id].get(local_instance_id, None)
        if instance_view is None:
            logger.debug(
                "instance view with local instance id",
                local_instance_id,
                "not found in unprocessed views",
            )
        return instance_view

    def add_view_to_instance(self, env_id: int, local_instance_id: int, global_instance_id: int):
        """
        Update instance associations and memory based on instance view information.

        This method handles the process of updating instance associations and memory based on instance view information.
        It ensures that the appropriate global instances are maintained or created and that their attributes are updated.

        Args:
            env_id (int): Identifier for the environment in which the update is performed.
            local_instance_id (int): Identifier for the local instance view within the specified environment.
            global_instance_id (int): Identifier for the global instance to which the local instance view will be associated.

        Note:
            - If the global instance with the given `global_instance_id` does not exist, a new global instance is created.
            - If the global instance already exists, the instance view is added to it, and its attributes are updated accordingly.

        Debugging:
            If the `debug_visualize` flag is enabled, the method generates visualizations for the instance association process
            and saves them to disk in the "instances" directory. Debug information is printed to the console.
        """

        # get instance view
        instance_view = self.get_local_instance_view(env_id, local_instance_id)

        # get global instance
        global_instance = self.instances[env_id].get(global_instance_id, None)
        if global_instance is None:
            global_instance = Instance(
                global_id=global_instance_id,
                score_aggregation_method=self.instance_view_score_aggregation_mode,
            )
            self.instances[env_id][global_instance_id] = global_instance
        global_instance.add_instance_view(instance_view)

        if self.debug_visualize:
            cat_id = int(instance_view.category_id)
            category_name = (
                f"cat_{instance_view.category_id}"
                if self.category_id_to_category_name is None
                else self.category_id_to_category_name[cat_id]
            )
            instance_write_path = os.path.join(
                self.log_dir, f"{global_instance_id}_{category_name}.png"
            )
            os.makedirs(instance_write_path, exist_ok=True)

            step = instance_view.timestep
            full_image = self.images[env_id][step]
            full_image = full_image.numpy().astype(np.uint8).transpose(1, 2, 0)
            # overlay mask on image
            mask = np.zeros(full_image.shape, full_image.dtype)
            mask[:, :] = (0, 0, 255)
            mask = cv2.bitwise_and(mask, mask, mask=instance_view.mask.astype(np.uint8))
            masked_image = cv2.addWeighted(mask, 1, full_image, 1, 0)
            cv2.imwrite(
                os.path.join(
                    instance_write_path,
                    f"step_{self.timesteps[env_id]}_local_id_{local_instance_id}.png",
                ),
                masked_image,
            )
            logger.debug(
                "mapping local instance id",
                local_instance_id,
                "to global instance id",
                global_instance_id,
            )

    def process_instances(
        self,
        instance_channels: Tensor,
        point_cloud: Tensor,
        image: Tensor,
        cam_to_world: Optional[Tensor] = None,
        semantic_channels: Optional[Tensor] = None,
        pose: Optional[Tensor] = None,
    ):
        """
        Process instance information across environments and associate instance views with global instances.

        This method processes instance information from instance channels, point cloud data, and images across different environments.
        It extracts and prepares instance views based on the provided data for each environment and associates them with global instances.

        Args:
            instance_channels (Tensor): Tensor containing instance segmentation channels for each environment.
            point_cloud (Tensor): Tensor containing point cloud data for each environment.
            image (Tensor): Tensor containing image data for each environment.
            semantic_channels (Optional[Tensor]): Tensor containing semantic segmentation channels for each environment, if available.

        Note:
            - Instance views are extracted and prepared for each environment based on the instance channels.
            - If semantic segmentation channels are provided, each instance view is associated with a semantic category.
            - Instance views are added to the list of unprocessed views for later association with specific instances.
            - After processing instance views for all environments, the method associates them with global instances using `associate_instances_to_memory()`.

        Debugging:
            If the `debug_visualize` flag is enabled, cropped images and visualization data are saved to disk.
        """
        instance_segs = instance_channels.argmax(dim=1).int()
        semantic_segs = None
        if semantic_channels is not None:
            semantic_segs = semantic_channels.argmax(dim=1).int()
        for env_id in range(self.num_envs):
            semantic_seg = None if semantic_segs is None else semantic_segs[env_id]
            self.process_instances_for_env(
                env_id,
                instance_segs[env_id],
                point_cloud[env_id],
                image[env_id],
                cam_to_world=cam_to_world[env_id] if cam_to_world is not None else None,
                semantic_seg=semantic_seg,
                pose=pose[env_id] if pose is not None else None,
            )
        self.associate_instances_to_memory()

    def _interpolate_image(self, image: Tensor, scale_factor: float = 1.0, mode: str = "nearest"):
        """
        Interpolates images by the specified scale_factor using the specific interpolation mode.

        This method uses `torch.nn.functional.interpolate` by temporarily adding batch dimension and channel dimension for 2D inputs.

        image (Tensor): image of shape [3, H, W] or [H, W]
        scale_factor (float): multiplier for spatial size
        mode: (str): algorithm for interpolation: 'nearest' (default), 'bicubic' or other interpolation modes at https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        """

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        image_downsampled = (
            torch.nn.functional.interpolate(
                image.unsqueeze(0).float(),
                scale_factor=scale_factor,
                mode=mode,
            )
            .squeeze()
            .squeeze()
            .bool()
        )
        return image_downsampled

    def filter_background_preds_using_clip(self, cropped_image, embedding):
        """
        Filter out background predictions using CLIP embeddings.
        """
        background_classes = (["wall", "ceiling", "floor", "others"],)
        clip_threshold = 1
        img_embeddings = embedding.unsqueeze(0)
        class_embeddings = torch.cat(
            [self.encoder.encode_text(bgc) for bgc in background_classes]
        ).to(cropped_image.device)
        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)

    def get_open_vocab_labels(self, cropped_image):
        img_embeddings = (
            self.encoder.encode_image(cropped_image).to(cropped_image.device).unsqueeze(0)
        )
        class_embeddings = torch.cat([self.encoder.encode_text(ovc) for ovc in self.open_vocab]).to(
            cropped_image.device
        )
        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        scores = (100.0 * img_embeddings @ class_embeddings.T).softmax(dim=-1).squeeze()
        return self.open_vocab[torch.argmax(scores).item()]

    def process_instances_for_env(
        self,
        env_id: int,
        instance_seg: Tensor,
        point_cloud: Tensor,
        image: Tensor,
        cam_to_world: Optional[Tensor] = None,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        semantic_seg: Optional[torch.Tensor] = None,
        background_class_labels: List[int] = [0],
        background_instance_labels: List[int] = [0],
        valid_points: Optional[Tensor] = None,
        pose: Optional[Tensor] = None,
        verbose: bool = False,
    ):
        """
        Process instance information in the current frame and add instance views to the list of unprocessed views for future association.

        This method processes instance information from instance segmentation, point cloud data, and images for a given environment.
        It extracts and prepares instance views based on the provided data and adds them to the list of unprocessed views for later association.

        Args:
            env_id (int): Identifier for the environment being processed.
            instance_seg (Tensor): [H, W] tensor of instance ids at each pixel
            point_cloud (Tensor): Point cloud data in world coordinates.
            image (Tensor): [3, H, W] RGB image
            cam_to_world: 4x4 camera_space_to_world transform
            instance_classes (Optional[Tensor]): [K,] class ids for each instance in instance seg
                class_int = instance_classes[instance_id]
            instance_scores (Optional[Tensor]): [K,] detection confidences for each instance in instance_seg
            semantic_seg (Optional[torch.Tensor]): Semantic segmentation tensor, if available.
            mask_out_object (bool): true if we want to save crops of just objects on black background; false otherwise
                # If false does it not save crops? Not black background?
            background_class_labels (List[int]): ids indicating background classes in semantic_seg. That view is not saved. (default = 0)
            background_instance_labels (List[int]): ids indicating background points in instance_seg. That view is not saved. (default = 0)
            valid_points (Tensor): [H, W] boolean tensor indicating valid points in the pointcloud
            pose: (Optional[Tensor]): base pose of the agent at this timestep
        Note:
            - The method creates instance views for detected instances within the provided data.
            - If a semantic segmentation tensor is provided, each instance is associated with a semantic category.
            - Instance views are added to the `unprocessed_views` dictionary for later association with specific instances.

        Debugging:
            If the `debug_visualize` flag is enabled, cropped images and visualization data are saved to disk.
        """
        # create a dict for mapping instance ids to categories
        instance_id_to_category_id = {}
        assert (
            image.shape[0] == 3
        ), "Ensure that RGB images are channels-first and in the right format."

        self.unprocessed_views[env_id] = {}
        # self.local_id_to_global_id_map[env_id] = {}
        # append image to list of images; move tensors to cpu to prevent memory from blowing up
        # TODO: This should probably be an option

        if self.images[env_id] is None:
            self.images[env_id] = image.unsqueeze(0).detach().cpu()
        else:
            self.images[env_id] = torch.cat(
                [self.images[env_id], image.unsqueeze(0).detach().cpu()], dim=0
            )
        if self.point_cloud[env_id] is None:
            self.point_cloud[env_id] = point_cloud.unsqueeze(0).detach().cpu()
        else:
            self.point_cloud[env_id] = torch.cat(
                [self.point_cloud[env_id], point_cloud.unsqueeze(0).detach().cpu()],
                dim=0,
            )

        # Valid points
        if valid_points is None:
            valid_points = torch.full_like(image[0], True, dtype=torch.bool, device=image.device)
        if self.du_scale != 1:
            valid_points_downsampled = interpolate_image(
                valid_points, scale_factor=1 / self.du_scale
            )
        else:
            valid_points_downsampled = valid_points

        t0 = timeit.default_timer()
        # unique instances
        instance_ids = torch.unique(instance_seg)
        for instance_id in instance_ids:
            t0 = timeit.default_timer()

            # skip background
            if instance_id in background_instance_labels:
                # print("Skipping background instance: ", instance_id, "in", background_instance_labels)
                continue
            # get instance mask
            instance_mask = instance_seg == instance_id
            # print (np.count_nonzero(instance_mask))
            category_id = None
            if instance_classes is not None:
                category_id = instance_classes[instance_id]
            elif semantic_seg is not None:
                assert (
                    instance_classes is None
                ), "cannot pass in both instance classes and semantic seg"
                # get semantic category
                category_id = semantic_seg[instance_mask].unique()
                category_id = category_id[0].item()

            # skip background
            if category_id is not None and category_id in background_class_labels:
                continue

            # detection score
            score = None
            if instance_scores is not None:
                score = instance_scores[instance_id]
            instance_id_to_category_id[instance_id] = category_id

            # get bounding box
            bbox = torch.stack(
                [
                    instance_mask.nonzero().min(dim=0)[0],
                    instance_mask.nonzero().max(dim=0)[0] + 1,
                ]
            )
            assert bbox.shape == (
                2,
                2,
            ), "Bounding box has extra dimensions - you have a problem with input instance image mask!"

            # TODO: If we use du_scale, we should apply this at the beginning to speed things up
            if self.du_scale != 1:
                instance_mask_downsampled = self._interpolate_image(
                    instance_mask, scale_factor=1 / self.du_scale
                )
                image_downsampled = self._interpolate_image(image, scale_factor=1 / self.du_scale)
            else:
                instance_mask_downsampled = instance_mask
                image_downsampled = image

            # Erode instance masks for point cloud
            # TODO: We can do erosion and masking on the downsampled/cropped image to avoid unnecessary computation
            if self.erode_mask_num_pix > 0:
                instance_mask_downsampled = dilate_or_erode_mask(
                    instance_mask_downsampled.unsqueeze(0),
                    radius=-self.erode_mask_num_pix,
                    num_iterations=self.erode_mask_num_iter,
                ).squeeze(0)
                instance_mask = dilate_or_erode_mask(
                    instance_mask.unsqueeze(0),
                    radius=-self.erode_mask_num_pix,
                    num_iterations=self.erode_mask_num_iter,
                ).squeeze(0)

            # Mask out the RGB image using the original detection mask
            if self.mask_cropped_instances:
                masked_image = image * instance_mask
            else:
                masked_image = image

            # get cropped image
            # p = self.padding_cropped_instances
            h, w = masked_image.shape[1:]
            cropped_image = self.get_cropped_image(image, bbox)
            instance_mask = self.get_cropped_image(instance_mask.unsqueeze(0), bbox)
            if self.mask_cropped_instances:
                cropped_image = cropped_image * instance_mask

            # get embedding
            if (
                self.encoder is not None
                and cropped_image.shape[0] * cropped_image.shape[1] > self.min_instance_points
                and cropped_image.shape[2] == 3
            ):

                # Compute semantic image features (e.g. SigLIP or CLIP)
                embedding = self.encoder.encode_image(cropped_image).to(cropped_image.device)

                # Get a separate set of visual, not semantic, features
                if hasattr(self.encoder, "get_visual_feat"):
                    visual_feat = self.encoder.get_visual_feat(cropped_image).to(
                        cropped_image.device
                    )
                else:
                    visual_feat = None
            else:
                continue

            # get point cloud
            point_mask_downsampled = instance_mask_downsampled & valid_points_downsampled
            point_cloud_instance = point_cloud[point_mask_downsampled]
            point_cloud_rgb_instance = image_downsampled.permute(1, 2, 0)[point_mask_downsampled]

            n_points = point_mask_downsampled.sum()
            n_mask = instance_mask_downsampled.sum()

            percent_points = n_mask / (instance_mask.shape[0] * instance_mask.shape[1])

            added = False
            if (
                n_mask >= self.min_pixels_for_instance_view
                and n_points > 1
                and percent_points > self.min_percent_for_instance_view
            ):
                bounds = get_bounds(point_cloud_instance)
                volume = float(box3d_volume_from_bounds(bounds).squeeze())

                if volume < float(self.min_instance_vol):
                    if verbose:
                        logger.info(
                            f"Skipping box with {n_points} points in cloud and {n_points} points in mask and {volume} volume",
                        )
                elif volume > float(self.max_instance_vol):
                    if verbose:
                        logger.info(
                            f"Skipping box with {n_points} points in cloud and {n_points} points in mask and {volume} volume",
                        )
                elif (
                    min(
                        bounds[0][1] - bounds[0][0],
                        bounds[1][1] - bounds[1][0],
                        bounds[2][1] - bounds[2][0],
                    )
                    < self.min_instance_thickness
                ):
                    if verbose:
                        logger.info(
                            f"Skipping a flat instance with {n_points} points",
                        )
                elif (bounds[2][0] + bounds[2][1]) / 2.0 < self.min_instance_height:
                    if verbose:
                        logger.info(
                            f"Skipping a instance with low height: {(bounds[2][0] + bounds[2][1]) / 2.0}",
                        )
                elif (bounds[2][0] + bounds[2][1]) / 2.0 > self.max_instance_height:
                    if verbose:
                        logger.info(
                            f"Skipping a instance with high height: {(bounds[2][0] + bounds[2][1]) / 2.0}",
                        )
                else:
                    # get instance view
                    instance_view = InstanceView(
                        bbox=bbox,
                        timestep=self.timesteps[env_id],
                        cropped_image=cropped_image,  # .cpu().numpy(),
                        embedding=embedding,
                        visual_feat=visual_feat,
                        mask=instance_mask,  # cpu().numpy().astype(bool),
                        point_cloud=point_cloud_instance,  # .cpu().numpy(),
                        point_cloud_rgb=point_cloud_rgb_instance,
                        cam_to_world=cam_to_world,
                        category_id=category_id,
                        score=score,
                        bounds=bounds,  # .cpu().numpy(),
                        pose=pose,
                        # open_vocab_label=open_vocab_label,
                    )
                    # append instance view to list of instance views
                    self.unprocessed_views[env_id][instance_id.item()] = instance_view
                    added = True
            else:
                logger.info(
                    f"Skipping a small instance with {n_mask} pixels and {n_points} points",
                )

            t1 = timeit.default_timer()
            if verbose:
                if added:
                    print(f"Added Instance {instance_id} took {t1-t0} seconds")
                else:
                    print(f"Skipped Instance {instance_id} took {t1-t0} seconds")

        # This timestep should be passable (e.g. for Spot we have a Datetime object)
        self.timesteps[env_id] += 1

    def get_unprocessed_instances_per_env(self, env_id: int):
        return self.unprocessed_views[env_id]

    def reset_for_env(self, env_id: int):
        self.instances[env_id] = {}
        self.images[env_id] = None
        self.point_cloud[env_id] = None
        self.unprocessed_views[env_id] = {}
        self.timesteps[env_id] = 0
        self.local_id_to_global_id_map[env_id] = {}

    def get_cropped_image(self, image, bbox):
        padding = self.crop_padding
        # image = instance_memory.images[0][iv.timestep]
        im_h = image.shape[1]
        im_w = image.shape[2]
        # bbox = iv.bbox
        x = bbox[0, 1]
        y = bbox[0, 0]
        w = bbox[1, 1] - x
        h = bbox[1, 0] - y
        x = 0 if (x - (padding - 1) * w / 2) < 0 else int(x - (padding - 1) * w / 2)
        y = 0 if (y - (padding - 1) * h / 2) < 0 else int(y - (padding - 1) * h / 2)
        y2 = im_h if y + int(h * padding) >= im_h else y + int(h * padding)
        x2 = im_w if x + int(w * padding) >= im_w else x + int(w * padding)
        cropped_image = (
            image[
                :,
                y:y2,
                x:x2,
            ].permute(1, 2, 0)
            # .byte()
        )
        return cropped_image

    def save(self, save_dir: Union[Path, str], env_id: int):
        pass

        # if self.debug_visualize:
        #     shutil.rmtree(self.log_dir, ignore_errors=True)

    def global_box_compression_and_nms(
        self,
        env_id: int,
        nms_thresh: Optional[float] = None,
        instance_box_compression_drop_prop: Optional[float] = None,
        instance_box_compression_resolution: Optional[float] = None,
    ) -> List[Instance]:

        if nms_thresh is None:
            nms_thresh = self.global_box_nms_thresh
        if instance_box_compression_drop_prop is None:
            instance_box_compression_drop_prop = self.instance_box_compression_drop_prop
        if instance_box_compression_resolution is None:
            instance_box_compression_resolution = self.instance_box_compression_resolution

        # Do NMS on global bboxes
        if nms_thresh > 0.0 and len(self.get_instances(env_id)) > 0:
            # Loop this since when we combine bboxes, they are now bigger.
            # And some bounding boxes might be inside the new larger bboxes
            num_global_inst = -1
            while len(self.get_instances(env_id)) != num_global_inst:
                if instance_box_compression_drop_prop > 0.0:
                    compress_instance_bounds_(
                        instances=self.get_instances(env_id),
                        drop_prop=instance_box_compression_drop_prop,
                        voxel_size=instance_box_compression_resolution,
                    )
                num_global_inst = len(self.get_instances(env_id))
                self.global_instance_nms(0, within_category=True, nms_iou_thresh=nms_thresh)

        # Compute tighter boxes by dropping lowest-weight points
        if instance_box_compression_drop_prop > 0.0 and len(self.get_instances(env_id)) > 0:
            compress_instance_bounds_(
                instances=self.get_instances(env_id),
                drop_prop=instance_box_compression_drop_prop,
                voxel_size=instance_box_compression_resolution,
            )
        return self.get_instances(env_id)

    def global_instance_nms(
        self,
        env_id: int,
        nms_iou_thresh: Optional[float] = None,
        within_category: bool = True,
    ) -> None:
        """
        Perform Non-Maximum Suppression (NMS) on global instances within the specified environment.

        This function performs NMS based on 3D bounding boxes and confidence scores. Instances that have a high overlap (IoU)
        and lower confidence scores are removed, and their instance views are added to the instance that is kept. The instances
        are reindexed after the NMS operation.

        Args:
            env_id: The environment ID where the global instances are located.
            nms_iou_thresh (Optional[float]): The IoU threshold for performing NMS.
                If not provided, the function uses the global_box_nms_thresh
                class variable as the threshold.
            within_category (bool): Only do NMS on objects of the same category

        Returns:
            None: This function modifies the internal state of the InstanceMemory object but does not return any value.

        Example usage:
            >>> instance_memory.global_instance_nms(env_id=0, nms_iou_thresh=0.5)

        Note:
            This function directly modifies the internal list of instances (`self.instances`) and
            reindexes them after the NMS operation.
        """
        if nms_iou_thresh is None:
            nms_iou_thresh = self.global_box_nms_thresh
        assert 0.0 < nms_iou_thresh and nms_iou_thresh <= 1.0, nms_iou_thresh
        if within_category:
            categories = torch.tensor(
                [inst.category_id for inst in self.get_instances_by_ids(env_id)]
            ).unique()
            for category_id in categories:
                category_id = int(category_id)
                ids_to_instances = self.get_ids_to_instances(env_id, category_id=category_id)
                ids, instances = list(ids_to_instances.keys()), list(ids_to_instances.values())
                instance_bounds = torch.stack([inst.bounds for inst in instances], dim=0)
                confidences = torch.stack([inst.score for inst in instances], dim=0)
                # Could implement a different NMS designed for partial views
                # Idea: Go over bounding boxes large-to-small
                # If intersection / small_volume > thresh -- assign small -> large
                confidences = box3d_volume_from_bounds(instance_bounds)  # Larger boxes get priority
                keep, vol, iou, assignments = box3d_sub_box_suppression(
                    instance_bounds,
                    confidence_score=confidences,
                    iou_threshold=nms_iou_thresh,
                )
                for keep_id, delete_ids in assignments.items():
                    # Map the sequential ids from box3d_nms to the instanceMemory global id
                    keep_id = ids[keep_id]
                    delete_ids = [ids[d_id] for d_id in delete_ids]
                    inst_to_keep = self.get_instance(env_id=env_id, global_instance_id=int(keep_id))
                    for delete_id in delete_ids:
                        inst_to_delete = self.pop_global_instance(
                            env_id, global_instance_id=int(delete_id), skip_reindex=True
                        )
                        for view in inst_to_delete.instance_views:
                            inst_to_keep.add_instance_view(view)
        else:
            global_instance_ids = self.get_global_instance_ids(env_id)
            instances = self.get_instances_by_ids(env_id, global_instance_ids)
            instance_bounds = torch.stack([inst.bounds for inst in instances], dim=0)
            instance_corners = get_box_verts_from_bounds(instance_bounds)
            confidences = torch.stack([inst.score for inst in instances], dim=0)
            keep, vol, iou, assignments = box3d_nms(
                instance_corners,
                confidence_score=confidences,
                iou_threshold=nms_iou_thresh,
            )
            for keep_id, delete_ids in assignments.items():
                inst_to_keep = self.get_instance(env_id=env_id, global_instance_id=int(keep_id))
                for delete_id in delete_ids:
                    inst_to_delete = self.pop_global_instance(
                        env_id, global_instance_id=int(delete_id), skip_reindex=True
                    )
                    for view in inst_to_delete.instance_views:
                        inst_to_keep.add_instance_view(view)
        self.reindex_global_instances(env_id)


def box3d_sub_box_suppression(bounds: Tensor, confidence_score: Tensor, iou_threshold: float = 0.3):
    """
    Non-max suppression where boxes inside other boxes get suppressed

    Args:
      bounds: (N, 3, 2) vertex coordinates. Must be in order specified by box3d_overlap
      confidence_score: (N,)
      iou_threshold: Suppress boxes whose IoU > iou_threshold

    Returns:
      keep, vol, iou, assignments
      keep: indexes into N of which bounding boxes to keep
      vol: (N, N) tensor of the volume of the intersecting convex shapes
      iov: (N, M) tensor of the intersection over union which is
          defined as: `iou = vol_int / (vol)`
      assignments: superbox_idx -> List[boxes_to_delete]
    """
    assert len(bounds) > 0, bounds.shape

    order = torch.argsort(confidence_score)

    (
        intersection,
        _,
        _,
    ) = box3d_intersection_from_bounds(bounds, bounds)
    volume = box3d_volume_from_bounds(bounds)
    intersection_over_volume = intersection / volume.unsqueeze(0)
    # There is a subtle possible bug here if we want to combine bboxes instead of just suppress
    # I.e. if box 1 contains most of box 8, but doesn't contain box1
    # and box 8 contains box 1
    # and box 1 is larger than box 8 larger than box 1
    # Then box 8 gets assigned to 1 and box 1 makes it through
    # Is this desired behavior?

    keep = []
    assignments = {}
    while len(order) > 0:

        idx = order[-1]  # Highest confidence (S)

        # push S in filtered predictions list
        keep.append(idx)

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # keep the boxes with IoU less than thresh_iou
        # _, iou = box3d_overlap(bounding_boxes[idx].unsqueeze(0), bounding_boxes[order])
        mask = intersection_over_volume[idx, order] < iou_threshold
        assignments[idx] = order[~mask]
        order = order[mask]

    return torch.tensor(keep), intersection, intersection_over_volume, assignments


def compress_instance_bounds_(
    instances: Sequence[Instance],
    drop_prop: float,
    voxel_size: float,
    min_voxels_to_compress: int = 3,
    min_vol: float = 1e-6,
):
    """Trailing _ in torch indicate in-place"""
    for instance in instances:
        reduced_points = drop_smallest_weight_points(
            instance.point_cloud,
            drop_prop=drop_prop,
            voxel_size=voxel_size,
            min_points_after_drop=min_voxels_to_compress,
        )
        new_bounds = get_bounds(reduced_points)
        if box3d_volume_from_bounds(new_bounds) >= min_vol:
            instance.bounds = new_bounds
    return instances
