# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pprint as pp

import cv2
import numpy as np
import torch
from lerobot.common.datasets.push_dataset_to_hub import dobbe_format

import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.app.lfd.policy_utils import load_policy, prepare_image, prepare_state
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder


class ROS2LfdLeader:
    """ROS2 version of leader for evaluating trained LfD policies with Stretch. To be used in conjunction with stretch_ros2_bridge server"""

    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        force_execute: bool = False,
        save_images: bool = False,
        teleop_mode: str = "base_x",
        record_success: bool = False,
        policy_path: str = None,
        policy_name: str = None,
        device: str = "cuda",
        depth_filter_k=None,
        disable_recording: bool = False,
    ):
        self.robot = robot

        self.save_images = save_images
        self.device = device
        self.policy_path = policy_path
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose

        # Save metadata to pass to recorder
        self.metadata = {
            "recording_type": "Policy evaluation",
            "user_name": user_name,
            "task_name": task_name,
            "env_name": env_name,
            "policy_name": policy_name,
            "policy_path": policy_path,
            "teleop_mode": self.teleop_mode,
            "backend": "ros2",
        }

        self._force = force_execute
        self._disable_recording = disable_recording
        self._recording = False or not self._disable_recording
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            data_dir, task_name, user_name, env_name, save_images, self.metadata
        )
        self._run_policy = False or self._force

        self.policy = load_policy(policy_name, policy_path, device)
        self.policy.reset()

    def ask_for_success(self) -> bool:
        """Ask the user if the episode was successful."""
        while True:
            logger.alert("Was the episode successful? (y/n)")
            key = cv2.waitKey(0)
            if key == ord("y"):
                return True
            elif key == ord("n"):
                return False

    def run(self, display_received_images: bool = False) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("dex_teleop_leader")
        try:
            while True:
                loop_timer.mark_start()

                # Get observation
                observation = self.robot.get_servo_observation()

                # Label joint states with appropriate format
                joint_states = {
                    k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
                }

                # Process images
                gripper_color_image = cv2.cvtColor(observation.ee_rgb, cv2.COLOR_RGB2BGR)
                gripper_depth_image = (
                    observation.ee_depth.astype(np.float32) * observation.ee_depth_scaling
                )
                head_color_image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
                head_depth_image = observation.depth.astype(np.float32) * observation.depth_scaling

                # Clip and normalize depth
                gripper_depth_image = dobbe_format.clip_and_normalize_depth(
                    gripper_depth_image, self.depth_filter_k
                )
                head_depth_image = dobbe_format.clip_and_normalize_depth(
                    head_depth_image, self.depth_filter_k
                )

                if display_received_images:

                    # change depth to be h x w x 3
                    combined = np.hstack((gripper_color_image / 255, gripper_depth_image / 4))

                    # Head images
                    head_combined = np.hstack((head_color_image / 255, head_depth_image / 4))

                    # Get the current height and width
                    (height, width) = combined.shape[:2]
                    (head_height, head_width) = head_combined.shape[:2]

                    # Calculate the aspect ratio
                    aspect_ratio = float(head_width) / float(head_height)

                    # Calculate the new height based on the aspect ratio
                    new_height = int(width / aspect_ratio)

                    head_combined = cv2.resize(
                        head_combined, (width, new_height), interpolation=cv2.INTER_LINEAR
                    )

                    # Combine both images from ee and head
                    combined = np.vstack((combined, head_combined))
                    cv2.imshow("Observed RGB/Depth Image", combined)

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
                        print("[LEADER] Recording stopped. Terminating.")
                        break
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
                        # Reset position
                        self.robot.arm_to(
                            [
                                0.0,  # base_x
                                0.9,  # lift
                                0.02,  # arm
                                0.0,  # wrist yaw, pitch, roll
                                -0.8,
                                0.0,
                            ],
                            gripper=self.robot._robot_model.GRIPPER_OPEN,
                        )
                else:
                    self._run_policy = True

                joint_actions = {}

                action = None
                if self._run_policy:
                    # Build state observations in correct format
                    observations = {
                        "observation.state": prepare_state(
                            joint_states, self.teleop_mode, self.device
                        ),
                        "observation.images.gripper": prepare_image(
                            gripper_color_image, self.device
                        ),
                        "observation.images.head": prepare_image(head_color_image, self.device),
                        "observation.images.gripper_depth": gripper_depth_image,
                        "observation.images.head_depth": head_depth_image,
                    }

                    # Send observation to policy
                    with torch.inference_mode():
                        raw_action = self.policy.select_action(observations)

                    # Get first batch
                    action = raw_action[0].tolist()

                    # Format raw_actions to order used for arm_to
                    # Order of raw actions is in dobbe_format.ACTION_ORDER
                    self.robot.arm_to(
                        [
                            action[0],  # base_x
                            action[3],  # lift
                            action[4],  # arm
                            action[7],  # yaw
                            action[6],  # pitch
                            action[5],  # roll
                        ],
                        gripper=action[8],  # gripper
                    )

                    # Label actions for saving
                    joint_actions = {
                        name: action[idx] for idx, name in enumerate(dobbe_format.ACTION_ORDER)
                    }
                else:
                    # If we aren't running the policy, what do we even need to do?
                    continue  # Skip the rest of the loop

                if self._recording:
                    if action is not None:
                        print("[LEADER] action=")
                        pp.pprint(action)

                    # Record episode if enabled
                    self._recorder.add(
                        ee_rgb=gripper_color_image,
                        ee_depth=gripper_depth_image,
                        xyz=np.array([0]),
                        quaternion=np.array([0]),
                        gripper=0,
                        ee_pos=np.array([0]),
                        ee_rot=np.array([0]),
                        observations=joint_states,
                        actions=joint_actions,
                        head_rgb=head_color_image,
                        head_depth=head_depth_image,
                    )
                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()

                # Stop condition for forced execution
                PITCH_STOP_THRESHOLD = -1.0

                stop = False
                PROGRESS_STOP_THRESHOLD = 0.95
                if len(action) == 10:
                    stop = action[9] > PROGRESS_STOP_THRESHOLD
                else:
                    stop = joint_actions["joint_wrist_pitch"] < PITCH_STOP_THRESHOLD

                if self._force and stop:
                    print(f"[LEADER] Stopping policy execution")
                    self._need_to_write = True

                    if self._disable_recording:
                        self._need_to_write = False
                        break

                if self._need_to_write:
                    if self.record_success:
                        success = self.ask_for_success()
                        print("[LEADER] Writing data to disk with success = ", success)
                        self._recorder.write(success=success)
                    else:
                        print("[LEADER] Writing data to disk.")
                        self._recorder.write()
                    self._need_to_write = False
                    if self._force:
                        break

        finally:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="", help="Robot IP address")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--user-name", type=str, default="default_user")
    parser.add_argument("-t", "--task-name", type=str, default="default_task")
    parser.add_argument("-e", "--env-name", type=str, default="default_env")
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force execute policy right away."
    )
    parser.add_argument("-d", "--data-dir", type=str, default="./data")
    parser.add_argument(
        "-s", "--save-images", action="store_true", help="Save raw images in addition to videos"
    )
    parser.add_argument("-P", "--send_port", type=int, default=4402, help="Port to send goals to.")
    parser.add_argument(
        "--teleop-mode",
        "--teleop_mode",
        type=str,
        default="base_x",
        choices=["stationary_base", "rotary_base", "base_x"],
    )
    parser.add_argument("--record-success", action="store_true", help="Record success of episode.")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to folder storing model weights"
    )
    parser.add_argument("--policy_name", type=str, required=True)
    parser.add_argument("--depth-filter-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun server for visualization."
    )
    parser.add_argument("--show-images", action="store_true", help="Show images received by robot.")
    args = parser.parse_args()

    # Parameters
    MANIP_MODE_CONTROLLED_JOINTS = dt_utils.get_teleop_controlled_joints(args.teleop_mode)
    parameters = get_parameters("default_planner.yaml")

    # Zmq client
    robot = HomeRobotZmqClient(
        robot_ip=args.robot_ip,
        send_port=args.send_port,
        parameters=parameters,
        manip_mode_controlled_joints=MANIP_MODE_CONTROLLED_JOINTS,
        enable_rerun_server=args.rerun,
    )
    robot.switch_to_manipulation_mode()
    robot.move_to_manip_posture()

    leader = ROS2LfdLeader(
        robot=robot,
        verbose=args.verbose,
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
        force_execute=args.force,
        save_images=args.save_images,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        policy_name=args.policy_name,
        policy_path=args.policy_path,
        device=args.device,
    )

    try:
        leader.run(display_received_images=args.show_images)
    except KeyboardInterrupt:
        pass

    robot.stop()
