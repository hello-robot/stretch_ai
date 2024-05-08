import cv2
import numpy as np


class Evaluator:
    """A basic class holding some overridable logic for evaluating input on sensors."""

    def __init__(self):
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
        color_image: np.ndarray,
        depth_image: np.ndarray,
        display_received_images: bool = True,
        **kwargs,
    ) -> dict:
        """Overwrite this to implement a simple loop operating on camera logic. Designed to decrease the amount of boilerplate code needed to implement a perception loop. Returns a results dict, which can be basically anything."""

        assert (self.camera_info is not None) and (
            self.depth_scale is not None
        ), "ERROR: YoloServoPerception: set_camera_parameters must be called prior to apply. self.camera_info or self.depth_scale is None"
        if display_received_images:
            cv2.imshow("Received RGB Image", color_image)
            cv2.imshow("Received Depth Image", depth_image)
            cv2.waitKey(1)

        results_dict = dict()
        return results_dict
