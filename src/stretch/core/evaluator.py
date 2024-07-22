import cv2
from typing import Any, Dict
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
