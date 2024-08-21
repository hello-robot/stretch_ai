#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
from typing import Any, Dict, Optional

import click
import numpy as np
from overrides import override

import stretch.motion.constants as constants
import stretch.utils.compression as compression
from stretch.core.server import BaseZmqServer
from stretch.motion import HelloStretchIdx
from stretch.simulation.stretch_mujoco import StretchMujocoSimulator
from stretch.utils.config import get_data_path
from stretch.utils.image import compute_pinhole_K, scale_camera_matrix


class MujocoZmqServer(BaseZmqServer):
    """Server for Mujoco simulation with ZMQ communication. This allows us to run the Mujoco simulation in the exact same way as we would run a remote ROS server on the robot, including potentially running it on a different machine or on the cloud. It requires:
    - Mujoco installation
    - Stretch_mujoco installation: https://github.com/hello-robot/stretch_mujoco/
    """

    def __init__(
        self, *args, scene_path: Optional[str] = None, simulation_rate: int = 200, **kwargs
    ):
        super(MujocoZmqServer, self).__init__(*args, **kwargs)
        if scene_path is None:
            scene_path = get_data_path("scene.xml")
        self.robot_sim = StretchMujocoSimulator(scene_path)
        self.simulation_rate = simulation_rate

        # Hard coded printout rates
        self.report_steps = 1000
        self.fast_report_steps = 10000
        self.servo_report_steps = 1000

        self._camera_data = None
        self._status = None

    def base_controller_at_goal(self):
        """Check if the base controller is at goal."""
        return True

    def get_joint_state(self):
        """Get the joint state of the robot."""
        status = self._status

        positions = np.zeros(constants.stretch_degrees_of_freedom)
        velocities = np.zeros(constants.stretch_degrees_of_freedom)
        efforts = np.zeros(constants.stretch_degrees_of_freedom)

        if status is None:
            return positions, velocities, efforts

        # Lift joint
        positions[HelloStretchIdx.LIFT] = status["lift"]["pos"]
        velocities[HelloStretchIdx.LIFT] = status["lift"]["vel"]

        # Arm joints
        positions[HelloStretchIdx.ARM] = status["arm"]["pos"]
        velocities[HelloStretchIdx.ARM] = status["arm"]["vel"]

        # Wrist roll joint
        positions[HelloStretchIdx.WRIST_ROLL] = status["wrist_roll"]["pos"]
        velocities[HelloStretchIdx.WRIST_ROLL] = status["wrist_roll"]["vel"]

        # Wrist yaw joint
        positions[HelloStretchIdx.WRIST_YAW] = status["wrist_yaw"]["pos"]
        velocities[HelloStretchIdx.WRIST_YAW] = status["wrist_yaw"]["vel"]

        # Wrist pitch joint
        positions[HelloStretchIdx.WRIST_PITCH] = status["wrist_pitch"]["pos"]
        velocities[HelloStretchIdx.WRIST_PITCH] = status["wrist_pitch"]["vel"]

        # Gripper joint
        positions[HelloStretchIdx.GRIPPER] = status["gripper"]["pos"]
        velocities[HelloStretchIdx.GRIPPER] = status["gripper"]["vel"]

        # Head pan joint
        positions[HelloStretchIdx.HEAD_PAN] = status["head_pan"]["pos"]
        velocities[HelloStretchIdx.HEAD_PAN] = status["head_pan"]["vel"]

        # Head tilt joint
        positions[HelloStretchIdx.HEAD_TILT] = status["head_tilt"]["pos"]
        velocities[HelloStretchIdx.HEAD_TILT] = status["head_tilt"]["vel"]

        return positions, velocities, efforts

    def get_base_pose(self) -> np.ndarray:
        """Base pose is the SE(2) pose of the base in world coords (x, y, theta)"""
        return self.robot_sim.get_base_pose()

    def get_ee_pose(self) -> np.ndarray:
        """EE pose is the 4x4 matrix of the end effector location in world coords"""
        return self.robot_sim.get_ee_pose()

    def get_head_camera_pose(self) -> np.ndarray:
        """Get the camera pose in world coords"""
        return self.robot_sim.get_link_pose("realsense")

    def get_ee_camera_pose(self) -> np.ndarray:
        """Get the end effector camera pose in world coords"""
        return self.robot_sim.get_link_pose("d405_cam")

    @override
    def get_control_mode(self) -> str:
        """Get the control mode of the robot."""
        return self.control_mode

    @override
    def start(self):
        self.robot_sim.start()  # This will start the simulation and open Mujoco-Viewer window
        super().start()
        while self.is_running():
            self._camera_data = self.robot_sim.pull_camera_data()
            self._status = self.robot_sim.pull_status()
            time.sleep(1 / self.simulation_rate)

    @override
    def handle_action(self, action: Dict[str, Any]):
        """Handle the action received from the client."""
        pass

    @override
    def get_full_observation_message(self) -> Dict[str, Any]:
        """Get the full observation message for the robot. This includes the full state of the robot, including images and depth images."""
        cam_data = self._camera_data
        if cam_data is None:
            return {}

        rgb = cam_data["cam_d435i_rgb"]
        depth = cam_data["cam_d435i_depth"]
        width, height = rgb.shape[:2]

        # Convert depth into int format
        depth = (depth * 1000).astype(np.uint16)

        # Get the joint state
        positions, _, _ = self.get_joint_state()

        # Make both into jpegs
        rgb = compression.to_jpg(rgb)
        depth = compression.to_jp2(depth)

        xyt = self.get_base_pose()

        # Get the other fields from an observation
        # rgb = compression.to_webp(rgb)
        message = {
            "rgb": rgb,
            "depth": depth,
            "camera_K": compute_pinhole_K(height, width, 69.4),
            "camera_pose": self.get_head_camera_pose(),
            "ee_pose": self.get_ee_pose(),
            "joint": positions,
            "gps": xyt[:2],
            "compass": xyt[2],
            "rgb_width": width,
            "rgb_height": height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": False,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self.base_controller_at_goal(),
        }
        return message

    @override
    def get_state_message(self) -> Dict[str, Any]:
        """Get the state message for the robot. This is a smalll message that includes floating point information and booleans like if the robot is homed."""
        q, dq, eff = self.get_joint_state()
        message = {
            "base_pose": self.get_base_pose(),
            "ee_pose": self.get_ee_pose(),
            "joint_positions": q,
            "joint_velocities": dq,
            "joint_efforts": eff,
            "control_mode": self.get_control_mode(),
            "at_goal": self.base_controller_at_goal(),
            "is_homed": True,
            "is_runstopped": False,
        }
        return message

    @override
    def get_servo_message(self) -> Dict[str, Any]:
        """Get messages for e2e policy learning and visual servoing. These are images and depth images, but lower resolution than the large full state observations, and they include the end effector camera."""

        cam_data = self._camera_data
        if cam_data is None:
            return {}

        head_color_image = cam_data["cam_d435i_rgb"]
        head_depth_image = cam_data["cam_d435i_depth"]
        ee_color_image = cam_data["cam_d405_rgb"]
        ee_depth_image = cam_data["cam_d405_depth"]

        # Adapt color so we can use higher shutter speed
        # TODO: do we need this? Probably not.
        # ee_color_image = adjust_gamma(ee_color_image, 2.5)

        ee_color_image, ee_depth_image = self._rescale_color_and_depth(
            ee_color_image, ee_depth_image, self.ee_image_scaling
        )
        head_color_image, head_depth_image = self._rescale_color_and_depth(
            head_color_image, head_depth_image, self.image_scaling
        )

        # Conversion
        ee_depth_image = (ee_depth_image * 1000).astype(np.uint16)
        head_depth_image = (head_depth_image * 1000).astype(np.uint16)

        # Compress the images
        compressed_ee_depth_image = compression.to_jp2(ee_depth_image)
        compressed_ee_color_image = compression.to_jpg(ee_color_image)
        compressed_head_depth_image = compression.to_jp2(head_depth_image)
        compressed_head_color_image = compression.to_jpg(head_color_image)

        # Get position info
        positions, _, _ = self.get_joint_state()

        # Get the camera matrices
        head_rgb_K = compute_pinhole_K(head_color_image.shape[0], head_color_image.shape[1], 69.4)
        head_dpt_K = compute_pinhole_K(head_depth_image.shape[0], head_depth_image.shape[1], 69.4)
        ee_rgb_K = compute_pinhole_K(ee_color_image.shape[0], ee_color_image.shape[1], 87.0)
        ee_dpt_K = compute_pinhole_K(ee_depth_image.shape[0], ee_depth_image.shape[1], 87.0)

        message = {
            "ee_cam/color_camera_K": scale_camera_matrix(ee_rgb_K, self.ee_image_scaling),
            "ee_cam/depth_camera_K": scale_camera_matrix(ee_dpt_K, self.ee_image_scaling),
            "ee_cam/color_image": compressed_ee_color_image,
            "ee_cam/depth_image": compressed_ee_depth_image,
            "ee_cam/color_image/shape": ee_color_image.shape,
            "ee_cam/depth_image/shape": ee_depth_image.shape,
            "ee_cam/image_scaling": self.ee_image_scaling,
            "ee_cam/depth_scaling": self.ee_depth_scaling,
            "ee_cam/pose": self.get_ee_camera_pose(),
            "ee/pose": self.get_ee_pose(),
            "head_cam/color_camera_K": scale_camera_matrix(head_rgb_K, self.image_scaling),
            "head_cam/depth_camera_K": scale_camera_matrix(head_dpt_K, self.image_scaling),
            "head_cam/color_image": compressed_head_color_image,
            "head_cam/depth_image": compressed_head_depth_image,
            "head_cam/color_image/shape": head_color_image.shape,
            "head_cam/depth_image/shape": head_depth_image.shape,
            "head_cam/image_scaling": self.image_scaling,
            "head_cam/depth_scaling": self.depth_scaling,
            "head_cam/pose": self.get_head_camera_pose(),
            "robot/config": positions,
        }
        return message

    @override
    def is_running(self) -> bool:
        """Check if the server is running. Will be used to make sure inner loops terminate.

        Returns:
            bool: True if the server is running, False otherwise."""
        return self.running and self.robot_sim.is_running()


@click.command()
@click.option("--send_port", default=4401, help="Port to send messages to clients")
@click.option("--recv_port", default=4402, help="Port to receive messages from clients")
@click.option("--send_state_port", default=4403, help="Port to send state-only messages to clients")
@click.option("--send_servo_port", default=4404, help="Port to send images for visual servoing")
@click.option("--use_remote_computer", default=True, help="Whether to use a remote computer")
@click.option("--verbose", default=False, help="Whether to print verbose messages")
@click.option("--image_scaling", default=0.5, help="Scaling factor for images")
@click.option("--ee_image_scaling", default=0.5, help="Scaling factor for end-effector images")
@click.option("--depth_scaling", default=0.001, help="Scaling factor for depth images")
@click.option(
    "--scene_path", default=None, help="Provide a path to mujoco scene file with stretch.xml"
)
def main(
    send_port: int,
    recv_port: int,
    send_state_port: int,
    send_servo_port: int,
    use_remote_computer: bool,
    verbose: bool,
    image_scaling: float,
    ee_image_scaling: float,
    depth_scaling: float,
    scene_path: str,
):
    server = MujocoZmqServer(
        send_port,
        recv_port,
        send_state_port,
        send_servo_port,
        use_remote_computer,
        verbose,
        image_scaling,
        ee_image_scaling,
        depth_scaling,
        scene_path=scene_path,
    )
    server.start()


if __name__ == "__main__":
    main()
