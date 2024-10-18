# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from stretch.core.abstract_perception import PerceptionModule


class SAMPerception(PerceptionModule):
    def __init__(self, model_type="vit_h", checkpoint_name: Optional[str] = None, verbose=False):
        super().__init__()
        self._verbose = verbose
        self.model_type = model_type
        if checkpoint_name is None:
            checkpoint_name = f"sam_{self.model_type}_4b8939.pth"
        self.checkpoint_name = checkpoint_name
        self.checkpoint_url = (
            f"https://dl.fbaipublicfiles.com/segment_anything/{self.checkpoint_name}"
        )

        self._download_checkpoint_if_needed()
        self._initialize_predictor()

    def _download_checkpoint_if_needed(self):
        """Try to download the checkpoint if it doesn't exist."""
        if not os.path.exists(self.checkpoint_name):
            print(f"Downloading {self.checkpoint_name}...")
            torch.hub.download_url_to_file(self.checkpoint_url, self.checkpoint_name)
            print("Download complete.")
        else:
            print(f"Checkpoint {self.checkpoint_name} already exists.")

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
        breakpoint()

        height, width, _ = rgb.shape

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
            raise ValueError("Either point_coords and point_labels, or box must be provided")

        return masks, scores, logits

    def is_semantic(self):
        """Whether the perception model is a semantic segmentation model."""
        return False

    def is_instance(self):
        """Whether the perception model is an instance segmentation model."""
        return True
