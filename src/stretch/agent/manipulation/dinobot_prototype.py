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
import open3d as o3d
import torch
from correspondences import find_correspondences, visualize_correspondences
from extractor import ViTExtractor
from scipy.spatial.transform import Rotation

import stretch.motion.constants as constants
from stretch.agent import RobotClient
from stretch.agent.manipulation.dinobot import (
    compute_error,
    extract_3d_coordinates,
    find_transformation,
)
from stretch.perception.detection.detic import DeticPerception
from stretch.visualization.urdf_visualizer import URDFVisualizer

DEBUG_VISUALIZATION = False


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
        self.urdf = URDFVisualizer()

    def get_correspondences(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        num_pairs: int = 20,
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

    def run(
        self,
        robot: RobotClient,
        demo: Demo,
        visualize: bool = False,
        apply_mask_callback=None,
        error_threshold: float = 0.01,
    ):
        """
        Run the Dinobot algorithm
        Args:
            robot (RobotClient): The robot client to use
            demo (Demo): The demonstration to replay
            visualize (bool): Whether to visualize the correspondences
        """
        print("Running Dinobot")
        error = 100000
        while error > error_threshold:
            servo = robot.get_servo_observation()
            ee_depth = servo.ee_depth.copy()
            ee_rgb = servo.ee_rgb.copy()
            if apply_mask_callback is not None:
                ee_rgb = apply_mask_callback(ee_rgb)
            if not isinstance(demo.bottleneck_image_rgb, type(None)):
                start = time.perf_counter()
                with torch.no_grad():
                    points1, points2 = self.get_correspondences(demo.bottleneck_image_rgb, ee_rgb)
                inf_ts = (time.perf_counter() - start) * 1000
                print(f"\n  current: {inf_ts} ms")
                if visualize:
                    if len(points1) == len(points2):
                        im1, im2 = visualize_correspondences(
                            points1, points2, demo.bottleneck_image_rgb, ee_rgb
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
                # Move the robot to the bottleneck pose
                error = self.move_to_bottleneck(robot, points1, points2, ee_depth, demo)
            else:
                print("No bottleneck image found")
        print("Dinobot finished")

    def move_robot(self, robot: RobotClient, R: np.ndarray, t: np.ndarray) -> float:
        """
        Move the robot to the bottleneck pose
        """
        model = robot.get_robot_model()
        T_d405 = self.urdf.get_transform_fk(
            robot.get_joint_positions(), "gripper_camera_color_optical_frame"
        )
        T_ee = self.urdf.get_transform_fk(robot.get_joint_positions(), "link_grasp_center")
        D405_target = np.eye(4)
        D405_target[:3, :3] = R
        D405_target[:3, 3] = t
        T_d405_target = np.matmul(T_d405, D405_target)

        T_d405_ee = np.matmul(np.linalg.inv(T_d405), T_ee)
        T_ee_target = np.matmul(T_d405_target, np.linalg.inv(T_d405_ee))

        DEBUG = True
        if DEBUG:

            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )

            T_d405_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            T_d405_frame.transform(T_d405)
            T_ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            T_ee_frame.transform(T_ee)

            T_target_d405_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            T_target_d405_frame.transform(T_d405_target)
            T_d405_target_blob = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            T_d405_target_blob.paint_uniform_color([0.15, 0.7, 0.15])  # Green for T_d405
            T_d405_target_blob.translate(T_d405_target[:3, 3])

            # T_ee_target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # T_ee_target_frame.transform(T_ee_target)
            # T_ee_target_blob = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            # T_ee_target_blob.paint_uniform_color([0.15, 0.15, 0.7])  # Blue for T_ee
            # T_ee_target_blob.translate(T_ee_target[:3, 3])

            # Create blobs at the origins of the coordinate frames
            origin_blob = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            origin_blob.paint_uniform_color([1, 0, 0])  # Red for origin

            T_d405_blob = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            T_d405_blob.paint_uniform_color([0, 1, 0])  # Green for T_d405

            T_ee_blob = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            T_ee_blob.paint_uniform_color([0, 0, 1])  # Blue for T_ee

            # Translate blobs to the origins of the coordinate frames
            T_d405_blob.translate(T_d405[:3, 3])
            T_ee_blob.translate(T_ee[:3, 3])

            # Add blobs to the visualization
            o3d.visualization.draw_geometries(
                [
                    origin_frame,
                    T_d405_frame,
                    T_ee_frame,
                    origin_blob,
                    T_d405_blob,
                    T_ee_blob,
                    T_target_d405_frame,
                    T_d405_target_blob,
                ]
                #  T_ee_target_frame,T_ee_target_blob]
            )

            # # Visualize the frames with labels
            # o3d.visualization.draw_geometries(
            #     [origin_frame, T_d405_frame, T_ee_frame]
            # )
        return

        T_ee_d405 = self.urdf_model.get_transform(
            "link_grasp_center", "gripper_camera_color_optical_frame"
        )

        T_ee_target = np.matmul(T_d405_target, np.linalg.inv(T_ee_d405))
        target_ee_pos = T_ee_target[:3, 3]
        target_ee_rot = T_ee_target[:3, :3]
        rot = Rotation.from_matrix(target_ee_rot)
        joint_state = robot.get_joint_positions().copy()
        target_joint_positions, _, _, success, _ = model.manip_ik_for_grasp_frame(
            target_ee_pos, rot.as_quat(), q0=joint_state
        )
        robot.switch_to_manipulation_mode()
        robot.arm_to(target_joint_positions, blocking=True, head=constants.look_at_ee)

    def move_to_bottleneck(
        self,
        robot: RobotClient,
        bottleneck_points: List,
        live_points: List,
        live_depth: np.ndarray,
        demo: Demo,
    ) -> float:
        """
        Compute the transformation to move the robot to the bottleneck pose
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

        invalid_depth_ids = []
        for i, point in enumerate(points1):
            if np.mean(point) == 0:
                invalid_depth_ids.append(i)
        for i, point in enumerate(points2):
            if np.mean(point) == 0:
                invalid_depth_ids.append(i)
        invalid_depth_ids = list(set(invalid_depth_ids))
        points1 = np.delete(points1, invalid_depth_ids, axis=0)
        points2 = np.delete(points2, invalid_depth_ids, axis=0)
        print(f" Number of valid correspondences: {len(points1)}")

        # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)
        r = Rotation.from_matrix(R)
        angles = r.as_euler("xyz")
        print(f"Camera frame to Rot: {R}, Trans: {t}")
        error = compute_error(points1, points2)
        print(f"Error: {error}")

        # Debug 3D Visualization
        if DEBUG_VISUALIZATION:
            unique_colors = generate_unique_colors(len(points1))
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.03, origin=[0, 0, 0]
            )

            transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.03, origin=t
            )
            transformed_frame.rotate(R, center=(0, 0, 0))

            # Bottleneck frame
            bottleneck_frame = o3d.geometry.PointCloud()
            bottleneck_frame.points = o3d.utility.Vector3dVector(
                bottleneck_xyz.reshape((bottleneck_xyz.shape[0] * bottleneck_xyz.shape[1], 3))
            )
            points1_frame = o3d.geometry.PointCloud()
            points1_frame.points = o3d.utility.Vector3dVector(points1)

            spheres = []
            for point, color in zip(points1, unique_colors):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.paint_uniform_color(color)
                sphere.translate(point)
                spheres.append(sphere)
            components = [origin_frame, bottleneck_frame, transformed_frame]
            components.extend(spheres)
            o3d.visualization.draw_geometries(components)

            # Live frame
            # live_frame = o3d.geometry.PointCloud()
            # live_frame.points = o3d.utility.Vector3dVector(live_xyz.reshape((live_xyz.shape[0]*live_xyz.shape[1],3)))
            # points2_frame = o3d.geometry.PointCloud()
            # points2_frame.points = o3d.utility.Vector3dVector(points2)

            # spheres = []
            # for point,color in zip(points2,unique_colors):
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            #     sphere.paint_uniform_color(color)
            #     sphere.translate(point)
            #     spheres.append(sphere)
            # components = [origin_frame,live_frame]
            # components.extend(spheres)
            # o3d.visualization.draw_geometries(components)

        self.move_robot(robot, R, t)
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


def generate_unique_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate N unique colors.
    Args:
        n (int): Number of unique colors to generate
    Returns:
        List[Tuple[int, int, int]]: List of RGB color tuples
    """
    colors = plt.cm.get_cmap("hsv", n)
    return [tuple(c for c in colors(i)[:3]) for i in range(n)]


if __name__ == "__main__":
    robot = RobotClient(robot_ip="10.0.0.2")
    dinobot = Dinobot()
    detic = DeticPerception()
    track_object_id = 41  # detic object id for cup

    # First frame is the bottleneck image for now
    bottleneck_image_rgb = robot.get_servo_observation().ee_rgb
    bottleneck_image_depth = robot.get_servo_observation().ee_depth
    bottleneck_image_camera_K = robot.get_servo_observation().ee_camera_K
    semantic, instance, task_observations = detic.predict(bottleneck_image_rgb)

    def apply_mask_callback(image: np.ndarray) -> np.ndarray:
        semantic, instance, task_observations = detic.predict(image)
        if track_object_id in task_observations["instance_classes"]:
            object_mask = semantic == track_object_id
            image[~object_mask] = [0, 0, 0]
        else:
            print(f"Object ID: {track_object_id} not found in the live image")
            breakpoint()
        return image

    if track_object_id in task_observations["instance_classes"]:
        object_mask = semantic == track_object_id
        demo = Demo(
            bottleneck_image_rgb, bottleneck_image_depth, bottleneck_image_camera_K, object_mask
        )
        input("Displace the object")
        dinobot.run(robot, demo, visualize=True, apply_mask_callback=apply_mask_callback)
    else:
        print(f"Object ID: {track_object_id} not found in the image")
