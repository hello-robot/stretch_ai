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

import json
from typing import Any, Dict, Optional, Tuple

import torch

import stretch.utils.logger as logger
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters, get_parameters
from stretch.perception.constants import RearrangeDETICCategories
from stretch.utils.config import get_full_config_path


class OvmmPerception:
    """
    Wrapper around DETIC for use in OVMM Agent.
    It performs some preprocessing of observations necessary for OVMM skills.
    It also maintains a list of vocabularies to use in segmentation and can switch between them at runtime.
    """

    def __init__(
        self,
        parameters: Parameters,
        gpu_device_id: int = 0,
        verbose: bool = False,
        confidence_threshold: float = 0.5,
        module_kwargs: Dict[str, Any] = {},
    ):
        self.parameters = parameters
        self._use_detic_viz = self.parameters.get("detection/use_detic_viz", False)
        self._detection_module = self.parameters.get("detection/module", "detic")
        self._vocabularies: Dict[int, RearrangeDETICCategories] = {}
        self._current_vocabulary: RearrangeDETICCategories = None
        self._current_vocabulary_id: int = None
        self.verbose = verbose
        if self._detection_module == "detic":
            # Lazy import
            from stretch.perception.detection.detic import DeticPerception

            self._segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=".",
                sem_gpu_id=gpu_device_id,
                verbose=verbose,
                confidence_threshold=confidence_threshold,
                **module_kwargs,
            )
        elif self._detection_module == "sam2":
            from stretch.perception.detection.sam2 import SAM2Perception

            self._segmentation = SAM2Perception()
        elif self._detection_module == "yolo":
            from stretch.perception.detection.yolo.yolo_perception import YoloPerception

            self._segmentation = YoloPerception(
                custom_vocabulary=".",
                sem_gpu_id=gpu_device_id,
                verbose=verbose,
                **module_kwargs,
            )
        else:
            raise NotImplementedError(f"Detection module {self._detection_module} not supported.")

    @property
    def current_vocabulary_id(self) -> int:
        return self._current_vocabulary_id

    @property
    def current_vocabulary(self) -> RearrangeDETICCategories:
        return self._current_vocabulary

    def update_vocabulary_list(self, vocabulary: RearrangeDETICCategories, vocabulary_id: int):
        """
        Update/insert a given vocabulary for the given ID.
        """
        self._vocabularies[vocabulary_id] = vocabulary

    def set_vocabulary(self, vocabulary_id: int):
        """
        Set given vocabulary ID to be the active vocabulary that the segmentation model uses.
        """
        vocabulary = self._vocabularies[vocabulary_id]
        self.segmenter_classes = ["."] + list(vocabulary.goal_id_to_goal_name.values()) + ["other"]
        self._segmentation.reset_vocab(self.segmenter_classes)

        self.vocabulary_name_to_id = {
            name: id for id, name in vocabulary.goal_id_to_goal_name.items()
        }
        self.vocabulary_id_to_name = vocabulary.goal_id_to_goal_name
        self.seg_id_to_name = dict(enumerate(self.segmenter_classes))
        self.name_to_seg_id = {v: k for k, v in self.seg_id_to_name.items()}

        self._current_vocabulary = vocabulary
        self._current_vocabulary_id = vocabulary_id

    def get_class_name_for_id(self, oid: int) -> Optional[str]:
        """return name of a class from a detection"""
        if isinstance(oid, torch.Tensor):
            oid = int(oid.item())
        if oid not in self._current_vocabulary.goal_id_to_goal_name:
            return None
        return self._current_vocabulary.goal_id_to_goal_name[oid]

    def get_class_id_for_name(self, name: str) -> Optional[int]:
        """return the id associated with a class"""
        if name in self._current_vocabulary.goal_name_to_goal_id:
            return self._current_vocabulary.goal_name_to_goal_id[name]
        return None

    def _process_obs(self, obs: Observations):
        """
        Process observations. Add pointers to objects and other metadata in segmentation mask.
        """
        obs.semantic[obs.semantic == 0] = self._current_vocabulary.num_sem_categories - 1
        obs.task_observations["recep_idx"] = self._current_vocabulary.num_sem_obj_categories + 1
        obs.task_observations["semantic_max_val"] = self._current_vocabulary.num_sem_categories - 1
        if obs.task_observations["start_recep_name"] is not None:
            obs.task_observations["start_recep_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["start_recep_name"]
            ]
        else:
            obs.task_observations["start_recep_name"] = None
        if obs.task_observations["place_recep_name"] is not None:
            obs.task_observations["end_recep_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["place_recep_name"]
            ]
        else:
            obs.task_observations["end_recep_name"] = None
        if obs.task_observations["object_name"] is not None:
            obs.task_observations["object_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["object_name"]
            ]
        else:
            obs.task_observations["object_goal"] = None

    def __call__(self, obs: Observations) -> Observations:
        return self.forward(obs)

    def predict_segmentation(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        base_pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Predict segmentation masks from RGB and depth images.

        Args:
            rgb (torch.Tensor): RGB image tensor
            depth (torch.Tensor): Depth image tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of semantic and instance segmentation masks
        """
        if base_pose is None:
            base_pose = torch.tensor([0.0, 0.0, 0.0])
        obs = Observations(rgb=rgb, depth=depth, gps=base_pose[0:2], compass=base_pose[2])
        obs = self.predict(obs)
        return obs.semantic, obs.instance, obs.task_observations

    def predict(
        self, obs: Observations, depth_threshold: Optional[float] = None, ee: bool = False
    ) -> Observations:
        """Run with no postprocessing. Updates observation to add semantics."""
        semantic, instance, task_observations = self._segmentation.predict(
            rgb=obs.rgb if not ee else obs.ee_rgb,
            depth=obs.depth if not ee else obs.ee_depth,
            depth_threshold=depth_threshold,
            draw_instance_predictions=self._use_detic_viz,
        )
        obs.semantic = semantic
        obs.instance = instance
        if obs.task_observations is None:
            obs.task_observations = task_observations
        else:
            obs.task_observations.update(task_observations)
        return obs

    def forward(self, obs: Observations, depth_threshold: float = 0.5) -> Observations:
        """
        Run segmentation model and preprocess observations for OVMM skills
        """
        obs = self.predict(obs, depth_threshold)
        self._process_obs(obs)
        return obs


def read_category_map_file(
    category_map_file: str,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Reads a category map file in JSON and extracts mappings between category names and category IDs.
    These mappings are also present in the episodes file but are extracted to use in a stand-alone manner.
    Returns object and receptacle mappings.
    """
    category_map_file = get_full_config_path(category_map_file)

    with open(category_map_file) as f:
        category_map = json.load(f)

    obj_name_to_id_mapping = category_map["obj_category_to_obj_category_id"]
    rec_name_to_id_mapping = category_map["recep_category_to_recep_category_id"]
    obj_id_to_name_mapping = {k: v for v, k in obj_name_to_id_mapping.items()}
    rec_id_to_name_mapping = {k: v for v, k in rec_name_to_id_mapping.items()}

    return obj_id_to_name_mapping, rec_id_to_name_mapping


def build_vocab_from_category_map(
    obj_id_to_name_mapping: Dict[int, str], rec_id_to_name_mapping: Dict[int, str]
) -> RearrangeDETICCategories:
    """
    Build vocabulary from category maps that can be used for semantic sensor and visualizations.
    """
    obj_rec_combined_mapping = {}
    for i in range(len(obj_id_to_name_mapping) + len(rec_id_to_name_mapping)):
        if i < len(obj_id_to_name_mapping):
            obj_rec_combined_mapping[i + 1] = obj_id_to_name_mapping[i]
        else:
            obj_rec_combined_mapping[i + 1] = rec_id_to_name_mapping[
                i - len(obj_id_to_name_mapping)
            ]
    vocabulary = RearrangeDETICCategories(obj_rec_combined_mapping, len(obj_id_to_name_mapping))
    return vocabulary


def create_semantic_sensor(
    parameters: Optional[Parameters] = None,
    category_map_file: Optional[str] = None,
    device_id: int = 0,
    verbose: bool = True,
    module_kwargs: Dict[str, Any] = {},
    config_path="default_planner.yaml",
    confidence_threshold: float = 0.5,
    **kwargs,
):
    """Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes."""
    if verbose:
        print("[PERCEPTION] Loading configuration")
    if parameters is None:
        parameters = get_parameters(config_path)
    if category_map_file is None:
        category_map_file = get_full_config_path(parameters["detection"]["category_map_file"])

    if verbose:
        logger.alert(
            "[PERCEPTION] Create and load vocabulary and perception model:",
            parameters["detection"]["module"],
        )
    semantic_sensor = OvmmPerception(
        parameters=parameters,
        gpu_device_id=device_id,
        verbose=verbose,
        confidence_threshold=confidence_threshold,
        module_kwargs=module_kwargs,
    )
    obj_name_to_id, rec_name_to_id = read_category_map_file(category_map_file)
    vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
    semantic_sensor.update_vocabulary_list(vocab, 0)
    semantic_sensor.set_vocabulary(0)
    return semantic_sensor
