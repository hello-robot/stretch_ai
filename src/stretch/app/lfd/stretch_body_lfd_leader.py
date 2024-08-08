# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pprint as pp
from typing import Optional

import cv2
import numpy as np
import torch
from lerobot.common.datasets.push_dataset_to_hub.dobbe_format import clip_and_normalize_depth

import stretch.utils.compression as compression
from stretch.app.lfd.policy_utils import load_policy, prepare_action_dict, prepare_observations
from stretch.core import Evaluator
from stretch.core.client import RobotClient
from stretch.utils.data_tools.record import FileDataRecorder
from stretch.utils.image import Camera


class StretchBodyLfdLeader(Evaluator):
    """Stretch body version of leader for evaluating trained LfD policies with stretch. To be used in conjunction with stretch.app.dex_teleop.follower"""

    def __init__(
        self,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        policy_path: str = None,
        policy_name: str = None,
        device: str = "cuda",
        force_record: bool = False,
        display_point_cloud: bool = False,
        save_images: bool = False,
        robot_ip: Optional[str] = None,
        recv_port: int = 4405,
        send_port: int = 4406,
        teleop_mode: str = None,
        depth_filter_k=None,
    ):
        super().__init__()
        self.camera = None

        self.display_point_cloud = display_point_cloud
        self.save_images = save_images
        self.device = device
        self.policy_path = policy_path
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k

        self.base_x_origin = None
        self.current_base_x = 0.0
        self.action_origin = None

        self.goal_send_socket = self._make_pub_socket(
            send_port, robot_ip=robot_ip, use_remote_computer=True
        )

        # Save metadata to pass to recorder
        self.metadata = {
            "recording_type": "Policy evaluation",
            "user_name": user_name,
            "task_name": task_name,
            "env_name": env_name,
            "policy_name": policy_name,
            "policy_path": policy_path,
            "teleop_mode": self.teleop_mode,
        }

        self._force = force_record
        self._recording = False or self._force
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            data_dir, task_name, user_name, env_name, save_images, self.metadata
        )
        self._run_policy = False

        self.policy = load_policy(policy_name, policy_path, device)
        self.policy.reset()

    def apply(self, message, display_received_images: bool = True) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""

        gripper_color_image = compression.from_webp(message["ee_cam/color_image"])
        gripper_depth_image = compression.unzip_depth(
            message["ee_cam/depth_image"], message["ee_cam/depth_image/shape"]
        )
        depth_camera_info = message["ee_cam/depth_camera_info"]
        depth_scale = message["ee_cam/depth_scale"]
        image_gamma = message["ee_cam/image_gamma"]
        image_scaling = message["ee_cam/image_scaling"]

        # Get head information from the message as well
        head_color_image = compression.from_webp(message["head_cam/color_image"])
        head_depth_image = compression.unzip_depth(
            message["head_cam/depth_image"], message["head_cam/depth_image/shape"]
        )
        head_depth_camera_info = message["head_cam/depth_camera_info"]
        head_depth_scale = message["head_cam/depth_scale"]

        if self.camera_info is None:
            self.set_camera_parameters(depth_camera_info, depth_scale)

        assert (self.camera_info is not None) and (
            self.depth_scale is not None
        ), "ERROR: YoloServoPerception: set_camera_parameters must be called prior to apply. self.camera_info or self.depth_scale is None"
        if self.camera is None:
            self.camera = Camera.from_K(
                self.camera_info["camera_matrix"],
                width=gripper_color_image.shape[1],
                height=gripper_color_image.shape[0],
            )

        # Rotate head camera images
        head_depth_image = cv2.rotate(head_depth_image, cv2.ROTATE_90_CLOCKWISE)
        head_color_image = cv2.rotate(head_color_image, cv2.ROTATE_90_CLOCKWISE)

        # Convert depth to meters
        gripper_depth_image = gripper_depth_image.astype(np.float32) * self.depth_scale
        head_depth_image = head_depth_image.astype(np.float32) * head_depth_scale

        # Clip and normalize depth
        gripper_depth_image = clip_and_normalize_depth(gripper_depth_image, self.depth_filter_k)
        head_depth_image = clip_and_normalize_depth(head_depth_image, self.depth_filter_k)

        if display_received_images:

            gripper_combined = np.hstack((gripper_depth_image / 255, gripper_color_image / 255))
            head_combined = np.hstack((head_depth_image / 255, head_color_image / 255))
            cv2.imshow("gripper", gripper_combined)
            cv2.imshow("head", head_combined)

        # Wait for spacebar to be pressed and start/stop recording
        # Spacebar is 32
        # Escape is 27
        key = cv2.waitKey(1)
        if key == 32:
            self._recording = not self._recording
            self.prev_goal_dict = None
            if self._recording:
                print("[LEADER] Recording started.")
            else:
                print("[LEADER] Recording stopped.")
                self._need_to_write = True
                if self._force:
                    # Try to terminate
                    print("[LEADER] Force recording done. Terminating.")
                    return None
        elif key == 27:
            if self._recording:
                self._need_to_write = True
            self._recording = False
            self.set_done()
            print("[LEADER] Recording stopped. Terminating.")
        elif key == ord("p"):
            self._run_policy = not self._run_policy
            if self._run_policy:
                # Reset base_x_origin
                self.base_x_origin = None
                self.action_origin = None
                self.policy.reset()
                print("[LEADER] Running policy!")
                self._recording = True
            else:
                self._recording = False
                self._need_to_write = True
                print("[LEADER] Stopping policy!")
        elif key == ord("r"):
            action_dict = {
                "joint_mobile_base_rotate_by": 0.0,
                "joint_lift": 0.9,
                "joint_arm_l0": 0.02,
                "joint_wrist_roll": 0.00,
                "joint_wrist_pitch": -0.80,
                "joint_wrist_yaw": 0.00,
                "stretch_gripper": 200.0,
            }
            self.goal_send_socket.send_pyobj(action_dict)
            return action_dict

        action_dict = {}
        action = []
        if self._run_policy:
            # Build state observations in correct format
            raw_state = message["robot/config"]

            if self.teleop_mode == "base_x":
                if self.base_x_origin is None:
                    print("Base_x reset!")
                    self.base_x_origin = raw_state["base_x"]

                # Calculate relative base_x
                self.current_base_x = raw_state["base_x"] - self.base_x_origin

                # Replace raw base_x with relative base_x
                raw_state["base_x"] = self.current_base_x

            observations = prepare_observations(
                raw_state,
                gripper_color_image,
                gripper_depth_image,
                head_color_image,
                head_depth_image,
                self.teleop_mode,
                self.device,
            )

            # Send observation to policy
            with torch.inference_mode():
                raw_action = self.policy.select_action(observations)

            # Get first batch
            action = raw_action[0].tolist()
            if self.action_origin is None:
                self.action_origin = action[0]

            # Format actions based on teleop_mode
            action_dict = prepare_action_dict(
                action, self.teleop_mode, self.current_base_x, self.action_origin
            )

        # Send action_dict to stretch follower
        self.goal_send_socket.send_pyobj(action_dict)

        if self._recording:
            print("[LEADER] action_dict =")
            pp.pprint(action_dict)

            # Record episode if enabled
            self._recorder.add(
                ee_rgb=gripper_color_image,
                ee_depth=gripper_depth_image,
                xyz=np.array([0]),
                quaternion=np.array([0]),
                gripper=0,
                ee_pos=np.array([0]),
                ee_rot=np.array([0]),
                observations=raw_state,
                actions=action_dict,
                head_rgb=head_color_image,
                head_depth=head_depth_image,
            )

        if self._need_to_write:
            print("[LEADER] Writing data to disk.")
            self._recorder.write()
            self._need_to_write = False

        return action_dict

    def __del__(self):
        self.goal_send_socket.close()
        if self._recording or self._need_to_write:
            self._recorder.write()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="192.168.1.15")
    parser.add_argument("-p", "--d405_port", type=int, default=4405)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--user-name", type=str, default="default_user")
    parser.add_argument("-t", "--task-name", type=str, default="default_task")
    parser.add_argument("-e", "--env-name", type=str, default="default_env")
    parser.add_argument("-r", "--replay", action="store_true", help="Replay a recorded session.")
    parser.add_argument("-f", "--force", action="store_true", help="Force data recording.")
    parser.add_argument("-d", "--data-dir", type=str, default="./data")
    parser.add_argument(
        "-s", "--save-images", action="store_true", help="Save raw images in addition to videos"
    )
    parser.add_argument("--display_point_cloud", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-P", "--send_port", type=int, default=4406, help="Port to send goals to.")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to folder storing model weights"
    )
    parser.add_argument("--policy_name", type=str, required=True)
    parser.add_argument("--teleop-mode", type=str, default="standard")
    parser.add_argument("--depth-filter-k", type=int, default=None)
    args = parser.parse_args()

    client = RobotClient(
        use_remote_computer=True,
        robot_ip=args.robot_ip,
        d405_port=args.d405_port,
        verbose=args.verbose,
    )

    # Create dex teleop leader - this will detect markers and send off goal dicts to the robot.
    evaluator = StretchBodyLfdLeader(
        robot_ip=args.robot_ip,
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
        policy_path=args.policy_path,
        policy_name=args.policy_name,
        device=args.device,
        force_record=args.force,
        display_point_cloud=args.display_point_cloud,
        save_images=args.save_images,
        send_port=args.send_port,
        teleop_mode=args.teleop_mode,
        depth_filter_k=args.depth_filter_k,
    )
    try:
        client.run(evaluator)
    except KeyboardInterrupt:
        pass
