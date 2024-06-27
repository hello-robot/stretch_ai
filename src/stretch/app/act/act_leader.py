import math
import os
import pprint as pp
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import zmq
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torchvision.transforms import v2

import stretch.utils.compression as compression
from stretch.app.act.policy_utils import load_policy
from stretch.core import Evaluator
from stretch.core.client import RobotClient
from stretch.utils.data_tools.record import FileDataRecorder
from stretch.utils.image import Camera
from stretch.utils.point_cloud import show_point_cloud


class ACTLeader(Evaluator):
    """A class for running an ACT model as leader with stretch."""

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
    ):
        super().__init__()
        self.camera = None

        self.display_point_cloud = display_point_cloud
        self.save_images = save_images
        self.device = device
        self.policy_path = policy_path

        self.goal_send_socket = self._make_pub_socket(
            send_port, robot_ip=robot_ip, use_remote_computer=True
        )

        self._force = force_record
        self._recording = False or self._force
        self._need_to_write = False
        self._recorder = FileDataRecorder(data_dir, task_name, user_name, env_name, save_images)
        self._run_policy = False

        self.policy = load_policy(policy_name, policy_path, device)
        self.policy.reset()

    def apply(self, message, display_received_images: bool = True) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""

        color_image = compression.from_webp(message["ee_cam/color_image"])
        depth_image = compression.unzip_depth(
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
                width=color_image.shape[1],
                height=color_image.shape[0],
            )

        # Convert depth to meters
        depth_image = depth_image.astype(np.float32) * self.depth_scale
        head_depth_image = head_depth_image.astype(np.float32) * head_depth_scale
        if self.display_point_cloud:
            print("depth scale", self.depth_scale)
            xyz = self.camera.depth_to_xyz(depth_image)
            show_point_cloud(xyz, color_image / 255, orig=np.zeros(3))

        if display_received_images:
            # change depth to be h x w x 3
            depth_image_x3 = np.stack((depth_image,) * 3, axis=-1)
            combined = np.hstack((color_image / 255, depth_image_x3 / 4))

            # Head images
            head_depth_image = cv2.rotate(head_depth_image, cv2.ROTATE_90_CLOCKWISE)
            head_color_image = cv2.rotate(head_color_image, cv2.ROTATE_90_CLOCKWISE)
            head_depth_image_x3 = np.stack((head_depth_image,) * 3, axis=-1)
            head_combined = np.hstack((head_color_image / 255, head_depth_image_x3 / 4))

            # # Get the current height and width
            # (height, width) = combined.shape[:2]
            # (head_height, head_width) = head_combined.shape[:2]

            # # Calculate the aspect ratio
            # aspect_ratio = float(head_width) / float(head_height)

            # # Calculate the new height based on the aspect ratio
            # new_height = int(width / aspect_ratio)

            # head_combined = cv2.resize(
            #     head_combined, (width, new_height), interpolation=cv2.INTER_LINEAR
            # )

            # # Combine both images from ee and head
            # combined = np.vstack((combined, head_combined))
            # cv2.imshow("Observed RGB/Depth Image", combined)
            cv2.imshow("gripper", color_image / 255)
            cv2.imshow("head", head_color_image / 255)

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

        if not self._recording:
            # Process WASD keys for motion
            if key == ord("w"):
                xyt = np.array([0.2, 0.0, 0.0])
            elif key == ord("a"):
                xyt = np.array([0.0, 0.0, -np.pi / 8])
            elif key == ord("s"):
                xyt = np.array([-0.2, 0.0, 0.0])
            elif key == ord("d"):
                xyt = np.array([0.0, 0.0, np.pi / 8])

        action_dict = {}
        action = []
        if self._run_policy:
            # Build state observations in correct format
            raw_state = message["robot/config"]
            state = np.array(
                [
                    raw_state["theta_vel"],
                    raw_state["joint_lift"],
                    raw_state["joint_arm_l0"],
                    raw_state["joint_wrist_roll"],
                    raw_state["joint_wrist_pitch"],
                    raw_state["joint_wrist_yaw"],
                    raw_state["stretch_gripper"],
                ]
            )

            state = torch.from_numpy(state)

            gripper_color_image_obs = torch.from_numpy(color_image)
            head_color_image_obs = torch.from_numpy(head_color_image)

            state = state.to(torch.float32)
            gripper_color_image_obs = gripper_color_image_obs.to(torch.float32) / 255
            gripper_color_image_obs = gripper_color_image_obs.permute(2, 0, 1)

            head_color_image_obs = head_color_image_obs.to(torch.float32) / 255
            head_color_image_obs = head_color_image_obs.permute(2, 0, 1)

            state = state.to(self.device, non_blocking=True)
            gripper_color_image_obs = gripper_color_image_obs.to(self.device, non_blocking=True)
            head_color_image_obs = head_color_image_obs.to(self.device, non_blocking=True)

            transforms = v2.Compose([v2.CenterCrop(320)])
            gripper_color_image_obs = transforms(gripper_color_image_obs)
            head_color_image_obs = transforms(head_color_image_obs)

            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            head_color_image_obs = head_color_image_obs.unsqueeze(0)
            gripper_color_image_obs = gripper_color_image_obs.unsqueeze(0)

            # Build observation dict for ACT
            observation = {
                "observation.state": state,
                "observation.images.gripper": gripper_color_image_obs,
                "observation.images.head": head_color_image_obs,
            }
            # Send observation to policy
            with torch.inference_mode():
                raw_action = self.policy.select_action(observation)

            # Get first batch
            action = raw_action[0].tolist()

            action_dict["joint_mobile_base_rotate_by"] = action[0]
            action_dict["joint_lift"] = action[1]
            action_dict["joint_arm_l0"] = action[2]
            action_dict["joint_wrist_roll"] = action[3]
            action_dict["joint_wrist_pitch"] = action[4]
            action_dict["joint_wrist_yaw"] = action[5]
            action_dict["stretch_gripper"] = action[6]

            action_dict["joint_mobile_base_rotate_by"] = 0.0

        # Send action_dict to stretch follower
        self.goal_send_socket.send_pyobj(action_dict)

        if self._recording:
            print("[LEADER] action_dict =")
            pp.pprint(action_dict)

            # Record episode if enabled
            self._recorder.add(
                ee_rgb=color_image,
                ee_depth=depth_image,
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
    args = parser.parse_args()

    client = RobotClient(
        use_remote_computer=True,
        robot_ip=args.robot_ip,
        d405_port=args.d405_port,
        verbose=args.verbose,
    )

    # Create dex teleop leader - this will detect markers and send off goal dicts to the robot.
    evaluator = ACTLeader(
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
    )
    try:
        client.run(evaluator)
    except KeyboardInterrupt:
        pass
