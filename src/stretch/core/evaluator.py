# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

from typing import Any, Dict

import cv2
import numpy as np

from .comms import ClientCommsNode


class Evaluator(ClientCommsNode):
    """A basic class holding some overridable logic for evaluating input on sensors."""

    def __init__(self):
        super(Evaluator, self).__init__()
        self.camera_info = None
        self.depth_scale = None
        self._done = False

    def set_done(self):
        """Tell the loop to safely exit. Should be handled in apply() if necessary and is used by the client."""
        self._done = True

    def is_done(self):
        """Should we close this loop?"""
        return self._done

    def set_camera_parameters(self, camera_info, depth_scale):
        """Set the camera parameters for the evaluator. This is necessary for the apply method to work properly."""
        self.camera_info = camera_info
        self.depth_scale = depth_scale

    def apply(
        self,
        message: Dict[str, Any],
        display_received_images: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Overwrite this to implement a simple loop operating on camera logic. Designed to decrease the amount of boilerplate code needed to implement a perception loop. Returns a results dict, which can be basically anything."""
        raise NotImplementedError
