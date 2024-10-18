# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from stretch.core import PerceptionModule


class MobileSAMPerception(PerceptionModule):
    def __init__(
        self,
        verbose: bool = True,
    ):
        super().__init__()
        self._verbose = verbose

        self.model_type = "vit_t"
        sam_checkpoint = "./weights/mobile_sam.pt"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[self.model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        self.mobile_sam = mobile_sam

        self.predictor = SamPredictor(mobile_sam)
        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        # masks = mask_generator.generate(<your_image>)

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
            rgb: image of shape (H, W, 3)
            depth: depth image of shape (H, W)
            depth_threshold: threshold for depth image to filter out objects
            draw_instance_predictions: whether to draw instance predictions

        Returns:
            semantic: semantic segmentation of shape (H, W)
            instance: instance segmentation of shape (H, W)
            metadata: metadata of the prediction (dict)
        """

        if self._verbose:
            print("SAM is segmenting the image...")

        # masks, scores, logits = self.segment(rgb)
        masks = self.mask_generator.generate(rgb)

        height, width, _ = rgb.shape
        semantic = np.zeros((height, width), dtype=np.uint8)
        instance = np.zeros((height, width), dtype=np.uint8)

        for i, mask in enumerate(masks):
            semantic[mask > 0] = i + 1
            instance[mask > 0] = i + 1

        task_observations = dict()
        task_observations["semantic_frame"] = semantic

        return semantic, instance, task_observations

    def segment(self, rgb, point_coords=None, point_labels=None, box=None, multimask_output=True):
        """Segment the image using loaded Segment Anything Model.

        Arguments:
            rgb: image of shape (H, W, 3)
            point_coords: coordinates of points to segment
            point_labels: labels of points to segment
            box: bounding box to segment
            multimask_output: whether to output multiple masks

        Returns:
            masks: masks of shape (N, H, W)
            scores: scores of the masks
            logits: logits of the masks
        """
        return self.predictor.segment(rgb, point_coords, point_labels, box, multimask_output)

    def is_instance(self) -> bool:
        """Return True if the perception module is an instance segmentation model."""
        return True

    def is_semantic(self) -> bool:
        """Return True if the perception module is a semantic segmentation model."""
        return True
