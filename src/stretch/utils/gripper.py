# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def get_gripper_aruco_detector() -> cv2.aruco.ArucoDetector:
    """Create an aruco detector preconfigured for the gripper AR markers to make it easier to track them."""
    aruco_parameters = cv2.aruco.DetectorParameters()
    aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    return aruco_detector


def detect_aruco_markers(
    image: np.ndarray, aruco_detector: cv2.aruco.ArucoDetector
) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    """Detect AR markers in an image."""
    corners, ids, _ = aruco_detector.detectMarkers(image)
    return corners, ids


class GripperArucoDetector:
    def __init__(self):
        self.aruco_detector = get_gripper_aruco_detector()

    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[Sequence[np.ndarray], np.ndarray]:
        """Detect AR markers in an image.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the corners of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        return detect_aruco_markers(image, self.aruco_detector)

    def detect_and_draw_aruco_markers(
        self, image: np.ndarray
    ) -> Tuple[Sequence[np.ndarray], np.ndarray, np.ndarray]:
        """Detect AR markers in an image and draw them.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the corners of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        corners, ids = self.detect_aruco_markers(image)
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return corners, ids, image

    def detect_aruco_centers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect AR markers in an image and return their centers.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the centers of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        corners, ids = self.detect_aruco_markers(image)
        centers = np.array([np.mean(c, axis=1) for c in corners])
        return centers, ids

    def detect_center(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get the center of the first detected AR marker in an image.

        Args:
            image: The image to detect the marker in.

        Returns:
            center: 2D array, The center point between the two finger AR markers.
        """
        centers, _ = self.detect_aruco_centers(image)
        if len(centers) < 2:
            return None
        center = (centers[0] + centers[1]) / 2
        return center[0]
