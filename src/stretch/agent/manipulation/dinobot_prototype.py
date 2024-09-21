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
import math
import os
import pickle
import time
from typing import List, Tuple

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
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

DEBUG = False


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


def Rz(theta):
    """
    Rotation matrix about z-axis
    """
    return np.matrix(
        [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
    )


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

    def run(
        self,
        robot: RobotClient,
        demo: Demo,
        visualize: bool = False,
        apply_mask_callback=None,
        error_threshold: float = 0.17,
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
                print("Extracting correspondences...")
                with torch.no_grad():
                    points1, points2 = self.get_correspondences(demo.bottleneck_image_rgb, ee_rgb)
                inf_ts = (time.perf_counter() - start) * 1000
                print(f"\n  Total correspondence extraction time: {inf_ts} ms")
                if visualize:
                    if len(points1) == len(points2):
                        im1, im2 = visualize_correspondences(
                            points1, points2, demo.bottleneck_image_rgb, ee_rgb
                        )
                        im1 = cv2.putText(
                            im1,
                            "Bottleneck Image",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        im2 = cv2.putText(
                            im2,
                            "Live Image",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        stacked_images = np.hstack((im1, im2))
                        # Enlarge the image for better visualization
                        stacked_images = cv2.resize(stacked_images, (0, 0), fx=2, fy=2)
                        cv2.imshow(
                            "Correspondences", cv2.cvtColor(stacked_images, cv2.COLOR_RGB2BGR)
                        )
                        while True:
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                            if cv2.getWindowProperty("Correspondences", cv2.WND_PROP_VISIBLE) < 1:
                                break
                        cv2.destroyAllWindows()
                    else:
                        print("No correspondences found")
                # Move the robot to the bottleneck pose
                error = self.move_to_bottleneck(robot, points1, points2, ee_depth, demo)
            else:
                print("No bottleneck image found")
        print("Dinobot finished")

    def move_robot(self, robot: RobotClient, R: np.ndarray, t: np.ndarray):
        """
        Move the robot to the bottleneck pose
        Args:
            robot (RobotClient): The robot client to use
            R (np.ndarray): The rotation matrix in D405 frame needed to be applied
            t (np.ndarray): The translation vector in D405 frame needed to be applied
        """
        model = robot.get_robot_model()
        T_d405 = self.urdf.get_transform_fk(
            robot.get_joint_positions(), "gripper_camera_color_optical_frame"
        )

        # Get the transformation from the bottleneck d405 frame to the target d405 frame
        T_ee = self.urdf.get_transform_fk(robot.get_joint_positions(), "link_grasp_center")
        D405_target = np.eye(4)
        D405_target[:3, :3] = R
        D405_target[:3, 3] = t
        T_d405_target = np.matmul(T_d405, D405_target)

        # Get the transformation from the target d405 frame to the target ee frame
        T_d405_ee = np.dot(np.linalg.inv(T_d405), T_ee)
        T_ee_target = T_d405_target.copy()

        # Extract the transformation to target ee frame
        R_d405_ee = T_d405_ee[:3, :3]
        t_d405_ee = T_d405_ee[:3, 3]
        T_ee_target[:3, 3] = T_ee_target[:3, 3] - t_d405_ee
        T_ee_target[:3, :3] = np.dot(T_ee_target[:3, :3], R_d405_ee)

        rerun_log(robot, T_d405_target)
        joint_state = robot.get_joint_positions().copy()

        if DEBUG:
            input("Press Enter to move the robot to the target pose")

        # Extract the target end-effector position and rotation
        # target_ee_pos = T_ee_target[:3, 3]
        # rot = Rotation.from_matrix(T_ee_target[:3, :3])

        try:
            # Compute the IK solution for the target end-effector position and rotation
            # target_joint_positions, _, _, success, _ = model.manip_ik_for_grasp_frame(
            #     target_ee_pos, rot.as_quat(), q0=joint_state
            # )

            # Compute the IK solution for the target D405 as custom ee frame
            target_d405_pos = T_d405_target[:3, 3]
            rot = Rotation.from_matrix(T_d405_target[:3, :3])
            joint_state = model._to_manip_format(joint_state)
            target_joint_positions, success, _ = model.manip_ik(
                pose_query=(target_d405_pos, rot.as_quat()),
                q0=joint_state,
                custom_ee_frame="gripper_camera_color_optical_frame",
            )

            # Move the robot to the target joint positions
            robot.switch_to_manipulation_mode()
            robot.arm_to(target_joint_positions, blocking=True, head=constants.look_at_ee)
        except Exception as e:
            print(f"Failed to move the robot to the target pose: {e}")

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
        print(f"Camera frame to Rot: {R}, Trans: {t}")
        error = compute_error(points1, points2)
        print(f"Error: {error}")

        # Rerun logging
        # Log correspondences 3d points from live image frame
        unique_colors = generate_unique_colors(len(points2))
        colors = np.array(unique_colors) * 255
        T_d405 = self.urdf.get_transform_fk(
            robot.get_joint_positions(), "gripper_camera_color_optical_frame"
        )
        base_xyt = robot.get_base_pose()
        base_4x4 = np.eye(4)
        base_4x4[:3, :3] = Rz(base_xyt[2])
        base_4x4[:2, 3] = base_xyt[:2]
        T_d405 = np.matmul(base_4x4, T_d405)
        rr.set_time_seconds("realtime", time.time())
        rr.log(
            "world/perceived_correspondences",
            rr.Points3D(points2, colors=colors.astype(np.uint8), radii=0.007),
        )
        rr.log(
            "world/perceived_correspondences",
            rr.Transform3D(translation=T_d405[:3, 3], mat3x3=T_d405[:3, :3], axis_length=0.00),
        )

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


class DemoRecorder(Demo):
    def __init__(
        self, image: np.ndarray, depth: np.ndarray, camera_K: np.ndarray, mask: np.ndarray
    ):
        super().__init__(image, depth, camera_K, mask)
        self.__first_frame = False
        self.trajectories = {}
        self.start_ts = None
        self.__id = 0
        self._delay = 2
        self.urdf = URDFVisualizer()

    def _step(self, ee_rgb: np.ndarray, ee_depth: np.ndarray, d405_frame: np.ndarray):
        if not self.__first_frame:
            self.__first_frame = True
            self.start_ts = time.time()
        self.trajectories[self.__id] = {
            "ee_rgb": ee_rgb,
            "ee_depth": ee_depth,
            "d405_frame": d405_frame,
        }
        self.__id += 1

    def save_demo(self, filepath: str = f"./demo_{time.strftime('%Y%m%d-%H%M%S')}"):
        demo_data = {
            "bottleneck_image_rgb": self.bottleneck_image_rgb,
            "bottleneck_image_depth": self.bottleneck_image_depth,
            "bottleneck_image_camera_K": self.bottleneck_image_camera_K,
            "object_mask": self.object_mask,
            "trajectories": self.trajectories,
        }

        with open(os.path.join(f"{filepath}.pkl"), "wb") as f:
            pickle.dump(demo_data, f)

    def collect_demo(
        self, robot: RobotClient, filepath: str = f"./demo_{time.strftime('%Y%m%d-%H%M%S')}"
    ):
        print("=================================================")
        print("Collect Demonstration through back driving on run stop")
        print("-------------------------------------------------")
        input(
            click.style(
                "Press Enter to start recording the demonstration frame by frame...",
                fg="yellow",
                bold=True,
            )
        )
        click.secho("Recording demonstration...", fg="green", bold=True)
        while True:
            servo = robot.get_servo_observation()
            ee_rgb = servo.ee_rgb.copy()
            ee_depth = servo.ee_depth.copy()
            d405_frame = self.urdf.get_transform_fk(
                robot.get_joint_positions(), "gripper_camera_color_optical_frame"
            )
            self._step(ee_rgb, ee_depth, d405_frame)
            if click.confirm("Record Next Frame?"):
                continue
            else:
                self.save_demo(filepath)
                click.secho(
                    f"Demonstration recording finished. N_Frames: {self.__id+1}",
                    fg="green",
                    bold=True,
                )
                break

    def replay_demo(self, robot: RobotClient):
        print("=================================================")
        print("            Replay Demonstration")
        print("-------------------------------------------------")
        for id in self.trajectories:
            print(f"Frame ID: {id}")
            self.move_to_d405_frame(robot, self.trajectories[id]["d405_frame"])
            time.sleep(self._delay)

    def move_to_d405_frame(self, robot: RobotClient, T_d405_target: np.ndarray):
        model = robot.get_robot_model()
        # Compute the IK solution for the target D405 as custom ee frame
        target_d405_pos = T_d405_target[:3, 3]
        rerun_log(robot, T_d405_target)
        rot = Rotation.from_matrix(T_d405_target[:3, :3])
        joint_state = robot.get_joint_positions().copy()
        joint_state = model._to_manip_format(joint_state)
        target_joint_positions, success, _ = model.manip_ik(
            pose_query=(target_d405_pos, rot.as_quat()),
            q0=joint_state,
            custom_ee_frame="gripper_camera_color_optical_frame",
        )

        # Move the robot to the target joint positions
        robot.switch_to_manipulation_mode()
        robot.arm_to(target_joint_positions, blocking=True, head=constants.look_at_ee)


def rerun_log(robot: RobotClient, T_d405_target: np.ndarray):
    """
    Rerun logging method
    """
    base_xyt = robot.get_base_pose()
    base_4x4 = np.eye(4)
    base_4x4[:3, :3] = Rz(base_xyt[2])
    base_4x4[:2, 3] = base_xyt[:2]

    # Rerun logging
    # Log the computed target bottleneck d405 and end-effector frames in the world frame
    rr.set_time_seconds("realtime", time.time())
    T_d405_target_world = np.matmul(base_4x4, T_d405_target)
    rr.log(
        "world/d405_bottleneck_frame/blob",
        rr.Points3D(
            [0, 0, 0],
            colors=[255, 255, 0, 255],
            labels="target_d405_bottleneck_frame",
            radii=0.01,
        ),
    )
    rr.log(
        "world/d405_bottleneck_frame",
        rr.Transform3D(
            translation=T_d405_target_world[:3, 3],
            mat3x3=T_d405_target_world[:3, :3],
            axis_length=0.3,
        ),
    )

    d405_bottleneck_arrow = rr.Arrows3D(
        origins=[0, 0, 0],
        vectors=[0, 0, 0.2],
        radii=0.0025,
        labels="d405_bottleneck_frame",
        colors=[128, 0, 128, 255],
    )
    rr.log("world/d405_bottleneck_frame/arrow", d405_bottleneck_arrow)

    rr.log(
        "world/ee_camera",
        rr.Points3D(
            [0, 0, 0],
            colors=[255, 255, 0, 255],
            labels="ee_camera_frame",
            radii=0.01,
        ),
    )

    current_d405_arrow = rr.Arrows3D(
        origins=[0, 0, 0],
        vectors=[0, 0, 0.5],
        radii=0.005,
        labels="d405_frame",
        colors=[255, 105, 180, 255],
    )
    rr.log("world/ee_camera/arrow", current_d405_arrow)

    rr.log(
        "world/ee",
        rr.Points3D(
            [0, 0, 0],
            colors=[0, 255, 255, 255],
            labels="ee_frame",
            radii=0.01,
        ),
    )


if __name__ == "__main__":
    DEBUG = False
    robot = RobotClient(robot_ip="10.0.0.2")
    dinobot = Dinobot()
    error_threshold = 0.17
    detic = DeticPerception()
    track_object_id = 41  # detic object id for cup

    print("\nFirst frame is the bottleneck image\n")
    print("=================================================")
    input(
        click.style(
            "Displace the object and backdrive robot to desired location and press Enter to collect demo:",
            fg="yellow",
            bold=True,
        )
    )

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

        # Collect demonstration
        demo = DemoRecorder(
            bottleneck_image_rgb, bottleneck_image_depth, bottleneck_image_camera_K, object_mask
        )
        demo.collect_demo(robot)
        input(
            click.style(
                "Displace the object and backdrive robot to another location and press Enter to try a grasp:",
                fg="yellow",
                bold=True,
            )
        )
        print("Visually servoing to the bottleneck pose...")
        # Visual Servo to bottleneck pose
        dinobot.run(
            robot,
            demo,
            visualize=DEBUG,
            apply_mask_callback=apply_mask_callback,
            error_threshold=error_threshold,
        )
        print("Replaying 3D trajectories...")
        # Replay demonstration
        demo.replay_demo(robot)
    else:
        print(f"Object ID: {track_object_id} not found in the image")
