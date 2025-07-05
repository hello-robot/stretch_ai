# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wget
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from stretch.core.abstract_perception import PerceptionModule
from stretch.perception.detection.utils import filter_depth, overlay_masks


class SAM2Perception(PerceptionModule):
    def __init__(
        self,
        custom_vocabulary: List[str] = "['', 'dog', 'grass', 'sky']",
        gpu_device_id: Optional[int] = None,
        configuration: str = "l",
        verbose=False,
    ):
        """Load trained SAM2 model for inference.

        Arguments:
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            gpu_device_id: GPU ID to load the model on, None for default
            configuration: size of SAM2 model, default is t, also supporting s, b+, l
            verbose: whether to print out debug information
        """
        # TODO: set model arg in config
        # You can and are recommended to run `download_ckpts.sh` in "." folder
        # to download sam weights.
        # Here we still provide weight checking and automatic downloading.
        # By default, we use the largest model now.
        base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
        if configuration == "t":
            checkpoint = "./sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            url = base_url + "sam2.1_hiera_tiny.pt"
        elif configuration == "s":
            checkpoint = "./sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            url = base_url + "sam2.1_hiera_small.pt"
        elif configuration == "b+":
            checkpoint = "./sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            url = base_url + "sam2.1_hiera_base_plus.pt"
        else:
            checkpoint = "./sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            url = base_url + "sam2.1_hiera_large.pt"
        if not os.path.exists(checkpoint):
            wget.download(url, out=checkpoint)

        self.sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        if not torch.cuda.is_available():
            if gpu_device_id is not None:
                print("Warning: CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif gpu_device_id is None:
            device = torch.device("cuda")
        elif gpu_device_id < 0 or gpu_device_id >= torch.cuda.device_count():
            print(
                f"Warning: Invalid GPU device ID {gpu_device_id}. Falling back to default CUDA device."
            )
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{gpu_device_id}")

        self.sam2_predictor = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        self._verbose = verbose

        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_predictor)

    def reset_vocab(self, new_vocab: List[str]):
        """Resets the vocabulary of the model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        self.custom_vocabulary = new_vocab

    def generate(self, image, min_area=500):  # TODO: figure out how to set min_area in config
        masks = self.mask_generator.generate(image)
        returned_masks = []
        for mask in masks:
            # filter out masks that are too small
            if mask["area"] >= min_area:
                returned_masks.append(mask["segmentation"])
        return returned_masks

    # Prompting SAM with detected boxes
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Get masks for all detected bounding boxes using SAM
        Arguments:
            image: image of shape (H, W, 3)
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            masks: masks of shape (N, H, W)
        """
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def predict(
        self,
        rgb=None,
        depth=None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Get masks using SAM
        Arguments:
            image: image of shape (H, W, 3)
        Returns:
            masks: masks of shape (N, H, W)
        """
        if self._verbose:
            print("SAM2 is segmenting the image...")

        height, width, _ = rgb.shape
        image = rgb
        image = np.array(image)
        if not image.dtype == np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)

        masks = np.array(self.generate(image))

        masks = sorted(masks, key=lambda x: np.count_nonzero(x), reverse=True)
        if depth_threshold is not None and depth is not None:
            masks = np.array([filter_depth(mask, depth, depth_threshold) for mask in masks])
        semantic_map, instance_map = overlay_masks(masks, np.zeros(len(masks)), (height, width))

        task_observations = dict()
        task_observations["instance_map"] = instance_map
        # random filling object classes -- right now using cups
        task_observations["instance_classes"] = np.full(len(masks), 31)
        task_observations["instance_scores"] = np.ones(len(masks))
        task_observations["semantic_frame"] = None
        return semantic_map.astype(int), instance_map.astype(int), task_observations

    def is_semantic(self):
        return True

    def is_instance(self):
        return True
