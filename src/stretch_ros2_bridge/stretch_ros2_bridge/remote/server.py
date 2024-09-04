#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 chris paxton for Hello Robot, under MIT license

from typing import Any, Dict

import click
import numpy as np
import rclpy
from overrides import override

import stretch.utils.compression as compression
import stretch.utils.logger as logger
from stretch.core.server import BaseZmqServer
from stretch.utils.image import adjust_gamma, scale_camera_matrix
from stretch_ros2_bridge.remote import StretchClient
from stretch_ros2_bridge.ros.map_saver import MapSerializerDeserializer


class ZmqServer(BaseZmqServer):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ROS2 client interface
        self.client = StretchClient(d405=True)

        # Map saver - write and load map information from SLAM
        self.map_saver = MapSerializerDeserializer()

    @override
    def is_running(self) -> bool:
        return self.is_running and rclpy.ok()

    @override
    def get_control_mode(self) -> str:
        """Get the current control mode of the robot. Can be navigation, manipulation, or none.

        Returns:
            str: The current control mode of the robot.
        """
        if self.client.in_manipulation_mode():
            control_mode = "manipulation"
        elif self.client.in_navigation_mode():
            control_mode = "navigation"
        else:
            control_mode = "none"
        return control_mode

    @override
    def get_full_observation_message(self) -> Dict[str, Any]:
        # get information
        # Still about 0.01 seconds to get observations
        obs = self.client.get_observation(compute_xyz=False)
        rgb, depth = obs.rgb, obs.depth
        width, height = rgb.shape[:2]

        # Convert depth into int format
        depth = (depth * 1000).astype(np.uint16)

        # Make both into jpegs
        rgb = compression.to_jpg(rgb)
        depth = compression.to_jp2(depth)

        # Get the other fields from an observation
        # rgb = compression.to_webp(rgb)
        data = {
            "rgb": rgb,
            "depth": depth,
            "camera_K": obs.camera_K.cpu().numpy(),
            "camera_pose": obs.camera_pose,
            "ee_pose": self.client.ee_pose,
            "joint": obs.joint,
            "gps": obs.gps,
            "compass": obs.compass,
            "rgb_width": width,
            "rgb_height": height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": self.client.last_motion_failed(),
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self.client.at_goal(),
        }
        return data

    @override
    def get_state_message(self) -> Dict[str, Any]:
        """Get the state message for the robot."""
        q, dq, eff = self.client.get_joint_state()
        message = {
            "base_pose": self.client.get_base_pose(),
            "ee_pose": self.client.ee_pose,
            "joint_positions": q,
            "joint_velocities": dq,
            "joint_efforts": eff,
            "control_mode": self.get_control_mode(),
            "at_goal": self.client.at_goal(),
            "is_homed": self.client.is_homed,
            "is_runstopped": self.client.is_runstopped,
        }
        return message

    @override
    def handle_action(self, action: Dict[str, Any]):
        """Handle an action from the client."""
        if "posture" in action:
            if action["posture"] == "manipulation":
                self.client.switch_to_busy_mode()
                self.client.move_to_manip_posture()
                self.client.switch_to_manipulation_mode()
            elif action["posture"] == "navigation":
                self.client.switch_to_busy_mode()
                self.client.move_to_nav_posture()
                self.client.switch_to_navigation_mode()
            else:
                print(
                    " - posture",
                    action["posture"],
                    "not recognized or supported.",
                )
        elif "control_mode" in action:
            if action["control_mode"] == "manipulation":
                self.client.switch_to_manipulation_mode()
                self.control_mode = "manipulation"
            elif action["control_mode"] == "navigation":
                self.client.switch_to_navigation_mode()
                self.control_mode = "navigation"
            else:
                print(
                    " - control mode",
                    action["control_mode"],
                    "not recognized or supported.",
                )
        elif "save_map" in action:
            self.client.save_map(action["save_map"])
        elif "load_map" in action:
            self.client.load_map(action["load_map"])
        elif "say" in action:
            # Text to speech from the robot, not the client/agent device
            self.text_to_speech.say_async(action["say"])
        elif "xyt" in action:
            if self.verbose:
                print(
                    "Is robot in navigation mode?",
                    self.client.in_navigation_mode(),
                )
                print(f"{action['xyt']} {action['nav_relative']} {action['nav_blocking']}")
            self.client.navigate_to(
                action["xyt"],
                relative=action["nav_relative"],
            )
        elif "base_velocity" in action:
            base_velocity_action = action["base_velocity"]
            if self.verbose:
                print(
                    f"Setting base velocity to translation={base_velocity_action['v']} and rotation={base_velocity_action['w']}"
                )
            if "v" in base_velocity_action:
                v = base_velocity_action["v"]
            if "w" in base_velocity_action:
                w = action["base_velocity"]["w"]
            self.client.nav.set_velocity(v, w)
        elif "joint" in action:
            # This allows for executing motor commands on the robot relatively quickly
            if self.verbose:
                print(f"Moving arm to config={action['joint']}")
            if "gripper" in action:
                gripper_cmd = action["gripper"]
            else:
                gripper_cmd = None
            if "head_to" in action:
                head_pan_cmd, head_tilt_cmd = action["head_to"]
            else:
                head_pan_cmd, head_tilt_cmd = None, None
            # Now send all command fields here
            self.client.arm_to(
                action["joint"],
                gripper=gripper_cmd,
                head_pan=head_pan_cmd,
                head_tilt=head_tilt_cmd,
                blocking=False,
            )
        elif "head_to" in action:
            # This will send head without anything else
            if self.verbose or True:
                print(f"Moving head to {action['head_to']}")
            self.client.head_to(
                action["head_to"][0],
                action["head_to"][1],
                blocking=False,
            )
        elif "gripper" in action and "joint" not in action:
            if self.verbose or True:
                print(f"Moving gripper to {action['gripper']}")
            self.client.manip.move_gripper(action["gripper"])
        else:
            logger.warning(" - action not recognized or supported.")
            logger.warning(action)

    def _get_ee_cam_message(self) -> Dict[str, Any]:
        # Read images from the end effector and head cameras
        ee_depth_image = self.client.ee_dpt_cam.get()
        ee_color_image = self.client.ee_rgb_cam.get()
        ee_color_image, ee_depth_image = self._rescale_color_and_depth(
            ee_color_image, ee_depth_image, self.ee_image_scaling
        )

        # Adapt color so we can use higher shutter speed
        ee_color_image = adjust_gamma(ee_color_image, 2.5)

        # Conversion
        ee_depth_image = (ee_depth_image * 1000).astype(np.uint16)

        # Compress the images
        compressed_ee_depth_image = compression.to_jp2(ee_depth_image)
        compressed_ee_color_image = compression.to_jpg(ee_color_image)

        ee_camera_pose = self.client.ee_camera_pose

        d405_output = {
            "ee_cam/color_camera_K": scale_camera_matrix(
                self.client.ee_rgb_cam.get_K(), self.ee_image_scaling
            ),
            "ee_cam/depth_camera_K": scale_camera_matrix(
                self.client.ee_dpt_cam.get_K(), self.ee_image_scaling
            ),
            "ee_cam/color_image": compressed_ee_color_image,
            "ee_cam/depth_image": compressed_ee_depth_image,
            "ee_cam/color_image/shape": ee_color_image.shape,
            "ee_cam/depth_image/shape": ee_depth_image.shape,
            "ee_cam/image_scaling": self.ee_image_scaling,
            "ee_cam/depth_scaling": self.ee_depth_scaling,
            "ee_cam/pose": ee_camera_pose,
        }
        return d405_output

    def get_servo_message(self) -> Dict[str, Any]:
        d405_output = self._get_ee_cam_message()

        obs = self.client.get_observation(compute_xyz=False)
        head_color_image, head_depth_image = self._rescale_color_and_depth(
            obs.rgb, obs.depth, self.image_scaling
        )
        head_depth_image = (head_depth_image * 1000).astype(np.uint16)
        compressed_head_depth_image = compression.to_jp2(head_depth_image)
        compressed_head_color_image = compression.to_jpg(head_color_image)

        message = {
            "ee/pose": self.client.ee_pose,
            "head_cam/color_camera_K": scale_camera_matrix(
                self.client.rgb_cam.get_K(), self.image_scaling
            ),
            "head_cam/depth_camera_K": scale_camera_matrix(
                self.client.dpt_cam.get_K(), self.image_scaling
            ),
            "head_cam/color_image": compressed_head_color_image,
            "head_cam/depth_image": compressed_head_depth_image,
            "head_cam/color_image/shape": head_color_image.shape,
            "head_cam/depth_image/shape": head_depth_image.shape,
            "head_cam/image_scaling": self.image_scaling,
            "head_cam/depth_scaling": self.depth_scaling,
            "head_cam/pose": self.client.head_camera_pose,
            "robot/config": obs.joint,
        }
        message.update(d405_output)
        return message


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--send_port", default=4401, help="Port to send observations to")
@click.option("--recv_port", default=4402, help="Port to receive actions from")
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
def main(
    send_port: int = 4401,
    recv_port: int = 4402,
    local: bool = False,
):
    rclpy.init()
    server = ZmqServer(
        send_port=send_port,
        recv_port=recv_port,
        use_remote_computer=(not local),
    )
    server.start()


if __name__ == "__main__":
    main()
