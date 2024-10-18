# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import timeit
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from stretch.core.abstract_perception import PerceptionModule
from stretch.utils.logger import Logger

logger = Logger(__name__)

checkpoint_names = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


class SAMPerception(PerceptionModule):
    def __init__(
        self,
        model_type="vit_b",
        checkpoint_name: Optional[str] = None,
        verbose: bool = True,
        default_spacing: int = 64,
    ):
        super().__init__()
        self._verbose = verbose
        self.model_type = model_type
        self.default_spacing = default_spacing
        if checkpoint_name is None:
            if model_type not in checkpoint_names:
                raise ValueError(
                    f"Model type {model_type} not supported. Choose from {checkpoint_names.keys()}"
                )
            checkpoint_name = checkpoint_names[model_type]
        self.checkpoint_name = checkpoint_name
        self.checkpoint_url = (
            f"https://dl.fbaipublicfiles.com/segment_anything/{self.checkpoint_name}"
        )

        self._download_checkpoint_if_needed()
        self._initialize_predictor()

    def _download_checkpoint_if_needed(self):
        """Try to download the checkpoint if it doesn't exist."""
        if not os.path.exists(self.checkpoint_name):
            logger.alert(f"Downloading {self.checkpoint_name}...")
            torch.hub.download_url_to_file(self.checkpoint_url, self.checkpoint_name)
            logger.alert("Download complete.")
        else:
            logger.info(f"Checkpoint {self.checkpoint_name} already exists.")

    def _initialize_predictor(self):
        """Initialize the SAM model and predictor."""
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_name)
        self.predictor = SamPredictor(sam)

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

        masks, scores, logits = self.segment(rgb)

        height, width, _ = rgb.shape
        semantic = np.zeros((height, width), dtype=np.uint8)
        instance = np.zeros((height, width), dtype=np.uint8)

        for i, mask in enumerate(masks):
            semantic[mask > 0] = i + 1
            instance[mask > 0] = i + 1

        task_observations = dict()
        task_observations["instance_scores"] = scores
        task_observations["instance_logits"] = logits
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

        t0 = timeit.default_timer()
        self.predictor.set_image(rgb)

        if point_coords is not None and point_labels is not None:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
        elif box is not None:
            masks, scores, logits = self.predictor.predict(
                box=box, multimask_output=multimask_output
            )
        else:
            # Box over the whole image
            point_coords, point_labels = generate_grid_points(rgb.shape[:2], self.default_spacing)
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
        t1 = timeit.default_timer()
        if self._verbose:
            print(f"Segmentation took {t1 - t0:.2f} seconds.")

        return masks, scores, logits

    def is_semantic(self):
        """Whether the perception model is a semantic segmentation model."""
        return False

    def is_instance(self):
        """Whether the perception model is an instance segmentation model."""
        return True


def generate_grid_points(image_shape, grid_spacing):
    """
    Generate a grid of points and labels for an image.

    Args:
    image_shape (tuple): Shape of the image (height, width).
    grid_spacing (int): Spacing between grid points.

    Returns:
    tuple: (point_coords, point_labels)
        point_coords: numpy array of shape (N, 2) containing x, y coordinates
        point_labels: numpy array of shape (N,) containing labels (all set to 1)
    """
    height, width = image_shape[:2]

    # Generate grid points
    x_coords = np.arange(grid_spacing // 2, width, grid_spacing)
    y_coords = np.arange(grid_spacing // 2, height, grid_spacing)

    xx, yy = np.meshgrid(x_coords, y_coords)

    # Reshape to (N, 2) array
    point_coords = np.column_stack((xx.ravel(), yy.ravel()))

    # Generate labels (all positive)
    point_labels = np.ones(point_coords.shape[0], dtype=int)

    return point_coords, point_labels
