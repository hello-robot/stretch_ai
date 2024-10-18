# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import timeit
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from torchvision.ops import nms

from stretch.core import PerceptionModule
from stretch.utils.logger import Logger

logger = Logger(__name__)


def non_maximum_suppression(results, iou_threshold=0.5):
    """Apply non-maximum suppression to the results.

    Arguments:
        results: list of results
        iou_threshold: IoU threshold for NMS

    Returns:
        List of results after NMS
    """
    if len(results) == 0:
        return results

    boxes = np.array([result["bbox"] for result in results]).astype(np.float32)
    scores = np.array([result["stability_score"] for result in results]).astype(np.float32)

    keep = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)

    return [results[i] for i in keep]


class MobileSAMPerception(PerceptionModule):
    def __init__(
        self,
        verbose: bool = True,
        min_score: float = 0.9,
        min_area: int = 100,
        io_threshold: float = 0.5,
    ):
        """
        Initialize the Mobile SAM perception module.

        Arguments:
            verbose: whether to print debug information
            min_score: minimum score for an object to be considered
            min_area: minimum area for an object to be considered
            io_threshold: IoU threshold for non-maximum suppression
        """
        super().__init__()
        self._verbose = verbose
        self.min_score = min_score
        self.min_area = min_area
        self.iou_threshold = io_threshold

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
            print("Mobile SAM is segmenting the image...")

        t0 = timeit.default_timer()
        # masks, scores, logits = self.segment(rgb)
        results = self.mask_generator.generate(rgb)
        results = non_maximum_suppression(results, self.iou_threshold)
        scores = []

        height, width, _ = rgb.shape
        semantic = np.zeros((height, width), dtype=np.uint8)
        instance = np.zeros((height, width), dtype=np.uint8)

        # Sort results by score from low to high
        results = sorted(results, key=lambda x: x["stability_score"], reverse=True)
        for i, result in enumerate(results):
            mask = result["segmentation"]
            score = result["stability_score"]

            if score < self.min_score:
                continue
            if np.sum(mask) < self.min_area:
                continue

            scores.append(score)
            semantic[mask > 0] = i + 1
            instance[mask > 0] = i + 1

            # if self._verbose:
            #     logger.info(f"Object {i + 1} has a stability score of {score}")

        # Second pass - remove small objects
        for i in range(1, instance.max() + 1):
            if np.sum(instance == i) < self.min_area:
                semantic[instance == i] = 0
                instance[instance == i] = 0

        task_observations = dict()
        task_observations["semantic_frame"] = semantic
        t1 = timeit.default_timer()

        if self._verbose:
            logger.info(f"Segmentation took {t1 - t0:.2f} seconds")

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
