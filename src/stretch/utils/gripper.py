from typing import Tuple

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect AR markers in an image."""
    corners, ids, _ = aruco_detector.detectMarkers(image)
    return corners, ids


class GripperArucoDetector:
    def __init__(self):
        self.aruco_detector = get_gripper_aruco_detector()

    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect AR markers in an image.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the corners of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        return detect_aruco_markers(image, self.aruco_detector)

    def detect_and_draw_aruco_markers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
