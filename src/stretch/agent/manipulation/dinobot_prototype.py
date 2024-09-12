# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import sys

sys.path.append("/home/hello-robot/repos/dino-vit-features")
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from correspondences import find_correspondences, visualize_correspondences
from extractor import ViTExtractor

from stretch.agent import RobotClient
from stretch.agent.manipulation.dinobot import (
    compute_error,
    extract_3d_coordinates,
    find_transformation,
)
from stretch.perception.detection.detic import DeticPerception


class Demo:
    """
    A demonstration from bottleneck pose
    """

    def __init__(
        self, image: np.ndarray, depth: np.ndarray, camera_K: np.ndarray, mask: np.ndarray
    ):
        """
        Initialize the Demo class
        Args:
            image (np.ndarray): The bottleneck image
            depth (np.ndarray): The bottleneck depth image
            camera_K (np.ndarray): The camera intrinsics
            mask (np.ndarray): The object mask
        """
        image[~mask] = [0, 0, 0]
        self.bottleneck_image_rgb = image.copy()
        self.bottleneck_image_depth = depth.copy()
        self.bottleneck_image_camera_K = camera_K.copy()
        self.object_mask = mask.copy()
        self.trajectories = {}

    @staticmethod
    def load_demo(self, path_to_demo_folder=None):
        # Load a demonstration from a folder containing data
        # TODO
        raise NotImplementedError


class Dinobot:
    """
    Dinobot is a class that uses DINO correspondences to move the robot to the bottleneck pose.
    And replay a demonstration.
    """

    def __init__(self, model_type: str = "dino_vits8", stride: int = 4):
        """
        Initialize the Dinobot class
        Args:
            model_type (str): The model type to use for feature extraction
            stride (int): The stride to use for feature extraction
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = ViTExtractor(
            model_type=model_type, stride=stride, device=self.device
        )

    def get_correspondences(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        num_pairs: int = 10,
        load_size: int = 224,
        layer: int = 9,
        facet: str = "key",
        bin: bool = True,
        thresh: float = 0.05,
    ) -> Tuple[List, List]:
        """
        Get correspondences key points between two images
        """
        points1, points2, image1_pil, image2_pil = find_correspondences(
            self.feature_extractor, image1, image2, num_pairs, load_size, layer, facet, bin, thresh
        )
        return points1, points2

    def run(self, robot: RobotClient, demo: Demo, visualize: bool = False):
        """
        Run the Dinobot algorithm
        Args:
            robot (RobotClient): The robot client to use
            demo (Demo): The demonstration to replay
            visualize (bool): Whether to visualize the correspondences
        """
        print("Running Dinobot")
        while True:
            servo = robot.get_servo_observation()
            ee_depth = servo.ee_depth
            if not isinstance(demo.bottleneck_image_rgb, type(None)):
                start = time.perf_counter()
                with torch.no_grad():
                    points1, points2 = self.get_correspondences(
                        demo.bottleneck_image_rgb, servo.ee_rgb
                    )
                    self.move_to_bottleneck(None, points1, points2, ee_depth, demo)
                inf_ts = (time.perf_counter() - start) * 1000
                print(f"\n  current: {inf_ts} ms")
                if visualize:
                    if len(points1) == len(points2):
                        im1, im2 = visualize_correspondences(
                            points1, points2, demo.bottleneck_image_rgb, servo.ee_rgb
                        )
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(im1)
                        axes[0].set_title("Bottleneck Image")
                        axes[0].axis("off")
                        axes[1].imshow(im2)
                        axes[1].set_title("Live Image")
                        axes[1].axis("off")
                        plt.show()
                    else:
                        print("No correspondences found")
            else:
                print("No bottleneck image found")

    def move_to_bottleneck(
        self,
        robot: RobotClient,
        bottleneck_points: List,
        live_points: List,
        live_depth: np.ndarray,
        demo: Demo,
    ) -> float:
        """
        Move the robot to the bottleneck pose
        Args:
            robot (RobotClient): The robot client to use
            bottleneck_points (List): The bottleneck points
            live_points (List): The live points
            live_depth (np.ndarray): The depth image
            demo (Demo): The demonstration to replay
        Returns:
            float: The error between the bottleneck and live points
        """
        # Given the pixel coordinates of the correspondences, and their depth values,
        # project the points to 3D space.
        bottleneck_xyz = depth_to_xyz(demo.bottleneck_image_depth, demo.bottleneck_image_camera_K)
        points1 = extract_3d_coordinates(bottleneck_points, bottleneck_xyz)
        live_xyz = depth_to_xyz(live_depth, demo.bottleneck_image_camera_K)
        points2 = extract_3d_coordinates(live_points, live_xyz)
        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)
        print(f"Robot needs to R: {R}, T: {t}")
        # TODO: Move robot
        error = compute_error(points1, points2)
        return error


def depth_to_xyz(depth: np.ndarray, camera_K: np.ndarray) -> np.ndarray:
    """get depth from numpy using simple pinhole camera model"""
    h, w = depth.shape
    indices = np.indices((h, w), dtype=np.float32).transpose(1, 2, 0)
    z = depth
    px, py = camera_K[0, 2], camera_K[1, 2]
    fx, fy = camera_K[0, 0], camera_K[1, 1]
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - px) * (z / fx)
    y = (indices[:, :, 0] - py) * (z / fy)
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


if __name__ == "__main__":
    robot = RobotClient(robot_ip="10.0.0.14")
    dinobot = Dinobot()
    detic = DeticPerception()
    track_object_id = 41  # detic object id for cup

    # First frame is the bottleneck image
    bottleneck_image_rgb = robot.get_servo_observation().ee_rgb
    bottleneck_image_depth = robot.get_servo_observation().ee_depth
    bottleneck_image_camera_K = robot.get_servo_observation().ee_camera_K
    semantic, instance, task_observations = detic.predict(bottleneck_image_rgb)
    if track_object_id in task_observations["instance_classes"]:
        object_mask = semantic == track_object_id
        demo = Demo(
            bottleneck_image_rgb, bottleneck_image_depth, bottleneck_image_camera_K, object_mask
        )
        dinobot.run(robot, demo, visualize=True)
    else:
        print(f"Object ID: {track_object_id} not found in the image")
