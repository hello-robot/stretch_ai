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


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PerceptionModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def is_semantic(self):
        """
        Whether the perception model is a semantic segmentation model.
        """
        pass

    @abstractmethod
    def is_instance(self):
        """
        Whether the perception model is an instance segmentation model.
        """
        pass

    def reset_vocab(self, vocab):
        """
        Reset the vocabulary of the perception model.

        Arguments:
            vocab: list of strings
        """
        pass

    @abstractmethod
    def predict(
        self,
        rgb=None,
        depth=None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run the perception model on the input images. Return semantic and instance labels for each point in the input image. Optionally filter out objects based on depth.

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
        pass
