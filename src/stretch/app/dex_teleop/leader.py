# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
import pprint as pp
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.app.dex_teleop.goal_from_teleop as gt
import stretch.app.dex_teleop.webcam_teleop_interface as wt
import stretch.motion.simple_ik as si
import stretch.utils.compression as compression
import stretch.utils.logger as logger
from stretch.core import Evaluator
from stretch.core.client import RobotClient
from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver
from stretch.utils.data_tools.record import FileDataRecorder
from stretch.utils.image import Camera
from stretch.utils.point_cloud import show_point_cloud

# Use Simple IK, if False use Pinocchio IK
use_simple_gripper_rules = False
use_gripper_center = True


class DexTeleopLeader(Evaluator):
    """A class for evaluating the DexTeleop system."""

    # Configurations for the head
    look_at_ee_cfg = np.array([-np.pi / 2, -np.pi / 4])
    look_front_cfg = np.array([0.0, math.radians(-30)])
    look_ahead_cfg = np.array([0.0, 0.0])
    look_close_cfg = np.array([0.0, math.radians(-45)])
    look_down_cfg = np.array([0.0, math.radians(-58)])

    # Configuration for IK
    _use_simple_gripper_rules = not use_gripper_center
    max_rotation_change = 0.5
    _ee_link_name = "link_grasp_center"
    debug_base_rotation = False

    def _create_ik_solver(
        self,
        urdf_path,
    ):
        self.manip_ik_solver = PinocchioIKSolver(
            urdf_path,
            self._ee_link_name,
            self._ik_joints_allowed_to_move,
        )

    def __init__(
        self,
        left_handed: bool = False,
        using_stretch2: bool = False,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        force_record: bool = False,
        display_point_cloud: bool = False,
        debug_aruco: bool = False,
        save_images: bool = False,
        robot_ip: Optional[str] = None,
        recv_port: int = 4405,
        send_port: int = 4406,
        teleop_mode: str = None,
        record_success: bool = False,
        platform: str = "linux",
    ):
        super().__init__()
        self.camera = None

        # TODO: fix these two things
        manipulate_on_ground = False
        slide_lift_range = False
        self.use_gripper_center = use_gripper_center
        self.display_point_cloud = display_point_cloud
        self.save_images = save_images
        self.teleop_mode = teleop_mode
        self.record_success = record_success
        self.platform = platform

        self.left_handed = left_handed
        self.using_stretch_2 = using_stretch2

        self.base_x_origin = None
        self.current_base_x = 0.0

        self.goal_send_socket = self._make_pub_socket(
            send_port, robot_ip=robot_ip, use_remote_computer=True
        )

        if self.teleop_mode == "base_x":
            self._ik_joints_allowed_to_move = [
                "joint_arm_l0",
                "joint_lift",
                "joint_wrist_yaw",
                "joint_wrist_pitch",
                "joint_wrist_roll",
                "joint_mobile_base_translation",
            ]
        else:
            self._ik_joints_allowed_to_move = [
                "joint_arm_l0",
                "joint_lift",
                "joint_wrist_yaw",
                "joint_wrist_pitch",
                "joint_wrist_roll",
                "joint_mobile_base_rotation",
            ]

        lift_middle = dt.get_lift_middle(manipulate_on_ground)
        center_configuration = dt.get_center_configuration(lift_middle)
        starting_configuration = dt.get_starting_configuration(lift_middle)

        if debug_aruco:
            logger.warning(
                "Debugging aruco markers. This displays an OpenCV UI which may make it difficult to enter commands. Do not use this option when doing data collection."
            )
        if left_handed:
            self.webcam_aruco_detector = wt.WebcamArucoDetector(
                tongs_prefix="left",
                visualize_detections=False,
                show_debug_images=debug_aruco,
                platform=platform,
            )
        else:
            self.webcam_aruco_detector = wt.WebcamArucoDetector(
                tongs_prefix="right",
                visualize_detections=False,
                show_debug_images=debug_aruco,
                platform=platform,
            )

        # Get Wrist URDF joint limits
        # rotary_urdf_file_name = "./stretch_base_rotation_ik_with_fixed_wrist.urdf"
        # rotary_urdf = load_urdf(rotary_urdf_file_name)
        translation_urdf_file_name = "./stretch_base_translation_ik_with_fixed_wrist.urdf"
        translation_urdf = dt_utils.load_urdf(translation_urdf_file_name)
        wrist_joints = ["joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"]
        self.wrist_joint_limits = {}
        for joint_name in wrist_joints:
            joint = translation_urdf.joint_map.get(joint_name, None)
            if joint is not None:
                lower = float(joint.limit.lower)
                upper = float(joint.limit.upper)
                self.wrist_joint_limits[joint.name] = (lower, upper)

        # rotary_urdf_with_wrist_file_name = "./stretch_base_rotation_ik.urdf"
        translation_urdf_with_wrist_file_name = "./stretch_base_translation_ik.urdf"
        print(self._ik_joints_allowed_to_move)
        self._create_ik_solver(translation_urdf_with_wrist_file_name)

        self.drop_extreme_wrist_orientation_change = True

        # Initialize the filtered wrist orientation that is used to
        # command the robot. Simple exponential smoothing is used to
        # filter wrist orientation values coming from the interface
        # objects.
        self.filtered_wrist_orientation = np.array([0.0, 0.0, 0.0])

        # Initialize the filtered wrist position that is used to command
        # the robot. Simple exponential smoothing is used to filter wrist
        # position values coming from the interface objects.
        self.filtered_wrist_position_configuration = np.array(
            [
                starting_configuration["joint_mobile_base_rotate_by"],
                starting_configuration["joint_lift"],
                starting_configuration["joint_arm_l0"],
            ]
        )

        self.prev_commanded_wrist_orientation = {
            "joint_wrist_yaw": None,
            "joint_wrist_pitch": None,
            "joint_wrist_roll": None,
        }

        if self.using_stretch_2:
            self.grip_range = dt.dex_wrist_grip_range
        else:
            self.grip_range = dt.dex_wrist_3_grip_range

        # This is the weight multiplied by the current wrist angle command when performing exponential smoothing.
        # 0.5 with 'max' robot speed was too noisy on the wrist
        self.wrist_orientation_filter = dt.exponential_smoothing_for_orientation

        # This is the weight multiplied by the current wrist position command when performing exponential smoothing.
        # commands before sending them to the robot
        self.wrist_position_filter = dt.exponential_smoothing_for_position

        self.print_robot_status_thread_timing = False
        self.debug_wrist_orientation = False

        self.max_allowed_wrist_yaw_change = dt.max_allowed_wrist_yaw_change
        self.max_allowed_wrist_roll_change = dt.max_allowed_wrist_roll_change

        # Initialize simple IK
        simple_ik = si.SimpleIK()
        if self._use_simple_gripper_rules:
            self.simple_ik = simple_ik
        else:
            self.simple_ik = None

        # Define the center position for the wrist that corresponds with
        # the teleop origin.
        self.center_wrist_position = simple_ik.fk_rotary_base(center_configuration)

        self.goal_from_markers = gt.GoalFromMarkers(
            dt.teleop_origin,
            self.center_wrist_position,
            slide_lift_range=slide_lift_range,
        )

        # Save metadata to pass to recorder
        self.metadata = {
            "recording_type": "Dex Teleop",
            "user_name": user_name,
            "task_name": task_name,
            "env_name": env_name,
            "left_handed": left_handed,
            "teleop_mode": teleop_mode,
            "backend": "stretch_body",
        }

        self._force = force_record
        self._recording = False or self._force
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            data_dir, task_name, user_name, env_name, save_images, self.metadata, fps=15
        )
        self.prev_goal_dict = None

    def apply(self, message, display_received_images: bool = True) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will run the aruco marker detection, predict a goal send that goal to the robot, and save everything to disk for learning."""

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

        # By default, no head or base commands
        head_cfg = None
        xyt = None

        # Wait for spacebar to be pressed and start/stop recording
        # Spacebar is 32
        # Escape is 27
        key = cv2.waitKey(1)
        if key == 32:
            self._recording = not self._recording
            self.prev_goal_dict = None
            if self._recording:
                # Reset base_x_origin
                self.base_x_origin = None
                print("[LEADER] Recording started.")
            else:
                print("[LEADER] Recording stopped.")
                self._need_to_write = True
                if self._force:
                    # Try to terminate
                    print("[LEADER] Force recording done. Terminating.")
                    return None
            head_cfg = self.look_ahead_cfg if not self._recording else self.look_at_ee_cfg
        elif key == 27:
            if self._recording:
                self._need_to_write = True
            self._recording = False
            self.set_done()
            print("[LEADER] Recording stopped. Terminating.")
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

        markers = self.webcam_aruco_detector.process_next_frame()

        # Set up commands to be sent to the robot
        goal_dict = self.goal_from_markers.get_goal_dict(markers)

        if goal_dict is not None:
            # Convert goal dict into a quaternion
            goal_dict = dt_utils.process_goal_dict(
                goal_dict, self.prev_goal_dict, self.use_gripper_center
            )
        else:
            # Goal dict that is not worth processing
            goal_dict = {"valid": False}
        # if head_cfg is not None:
        #     goal_dict["head_config"] = head_cfg
        if xyt is not None:
            goal_dict["move_xyt"] = xyt

        # Process incoming state based on teleop mode
        raw_state_received = message["robot/config"]
        goal_dict["current_state"] = dt_utils.format_state(raw_state_received, self.teleop_mode)

        if goal_dict["valid"]:
            # Process teleop gripper goal to goal joint configurations using IK
            goal_configuration = self.get_goal_joint_config(**goal_dict)

            # TODO temporary implementation of teleop mode filtering
            if self.teleop_mode == "stationary_base":
                goal_configuration["joint_mobile_base_rotate_by"] = 0.0
            elif self.teleop_mode == "base_x":
                # Override with reset base_x
                goal_dict["current_state"]["base_x"] = self.current_base_x

            if self._recording:
                print("[LEADER] goal_dict =")
                pp.pprint(goal_configuration)

            # Record episode if enabled
            if self._recording and self.prev_goal_dict is not None:
                self._recorder.add(
                    color_image,
                    depth_image,
                    goal_dict["relative_gripper_position"],
                    goal_dict["relative_gripper_orientation"],
                    goal_dict["grip_width"],
                    head_rgb=head_color_image,
                    head_depth=head_depth_image,
                    observations=goal_dict["current_state"],
                    actions=goal_configuration,
                    ee_pos=message["robot/ee_position"],
                    ee_rot=message["robot/ee_rotation"],
                )

            # Send goal joint configuration to robot
            self.goal_send_socket.send_pyobj(goal_configuration)
        else:
            # Send original goal_dict with valid=false
            self.goal_send_socket.send_pyobj(goal_dict)

        self.prev_goal_dict = goal_dict

        if self._need_to_write:
            if self.record_success:
                success = self.ask_for_success()
                print("[LEADER] Writing data to disk with success = ", success)
                self._recorder.write(success=success)
            else:
                print("[LEADER] Writing data to disk.")
                self._recorder.write()
            self._need_to_write = False
        return goal_dict

    def ask_for_success(self) -> bool:
        """Ask the user if the episode was successful."""
        while True:
            logger.alert("Was the episode successful? (y/n)")
            key = cv2.waitKey(0)
            if key == ord("y"):
                return True
            elif key == ord("n"):
                return False

    def get_goal_joint_config(
        self,
        grip_width,
        wrist_position: np.ndarray,
        gripper_orientation: np.ndarray,
        current_state,
        relative: bool = False,
        verbose: bool = False,
        **config,
    ):
        # Process goal dict in gripper pose format to full joint configuration format with IK
        # INPUT: wrist_position
        if (
            "use_gripper_center" in config
            and config["use_gripper_center"] != self.use_gripper_center
        ):
            raise RuntimeError("leader and follower are not set up to use the same target poses.")

        # Use Simple IK to find configurations for the
        # mobile base angle, lift distance, and arm
        # distance to achieve the goal wrist position in
        # the world frame.
        if self._use_simple_gripper_rules:
            new_goal_configuration = self.simple_ik.ik_rotary_base(wrist_position)
        else:
            res, success, info = self.manip_ik_solver.compute_ik(
                wrist_position,
                gripper_orientation,
                q_init=current_state,
                ignore_missing_joints=True,
            )
            new_goal_configuration = self.manip_ik_solver.q_array_to_dict(res)
            if not success:
                print("!!! BAD IK SOLUTION !!!")
                new_goal_configuration = None
            if verbose:
                pp.pp(new_goal_configuration)

        if new_goal_configuration is None:
            print(
                f"WARNING: IK failed to find a valid new_goal_configuration so skipping this iteration by continuing the loop. Input to IK: wrist_position = {wrist_position}, Output from IK: new_goal_configuration = {new_goal_configuration}"
            )
        else:
            if self.teleop_mode == "base_x":
                new_wrist_position_configuration = np.array(
                    [
                        new_goal_configuration["joint_mobile_base_translation"],
                        new_goal_configuration["joint_lift"],
                        new_goal_configuration["joint_arm_l0"],
                    ]
                )
            else:
                new_wrist_position_configuration = np.array(
                    [
                        new_goal_configuration["joint_mobile_base_rotation"],
                        new_goal_configuration["joint_lift"],
                        new_goal_configuration["joint_arm_l0"],
                    ]
                )
            # Use exponential smoothing to filter the wrist
            # position configuration used to command the
            # robot.
            self.filtered_wrist_position_configuration = (
                (1.0 - self.wrist_position_filter) * self.filtered_wrist_position_configuration
            ) + (self.wrist_position_filter * new_wrist_position_configuration)

            new_goal_configuration["joint_lift"] = self.filtered_wrist_position_configuration[1]
            new_goal_configuration["joint_arm_l0"] = self.filtered_wrist_position_configuration[2]

            if self._use_simple_gripper_rules:
                self.simple_ik.clip_with_joint_limits(new_goal_configuration)

            #################################

            #################################
            # INPUT: grip_width between 0.0 and 1.0

            if (grip_width is not None) and (grip_width > -1000.0):
                new_goal_configuration["stretch_gripper"] = self.grip_range * (grip_width - 0.5)

            ##################################################
            # INPUT: x_axis, y_axis, z_axis

            if self._use_simple_gripper_rules:
                # Use the gripper pose marker's orientation to directly control the robot's wrist yaw, pitch, and roll.
                r = Rotation.from_quat(gripper_orientation)
                if relative:
                    print("!!! relative rotations not yet supported !!!")
                wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_position(r)
            else:
                wrist_yaw = new_goal_configuration["joint_wrist_yaw"]
                wrist_pitch = new_goal_configuration["joint_wrist_pitch"]
                wrist_roll = new_goal_configuration["joint_wrist_roll"]

            if self.debug_wrist_orientation:
                print("___________")
                print(
                    "wrist_yaw, wrist_pitch, wrist_roll = {:.2f}, {:.2f}, {:.2f} deg".format(
                        (180.0 * (wrist_yaw / np.pi)),
                        (180.0 * (wrist_pitch / np.pi)),
                        (180.0 * (wrist_roll / np.pi)),
                    )
                )

            limits_violated = False
            lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_yaw"]
            if (wrist_yaw < lower_limit) or (wrist_yaw > upper_limit):
                limits_violated = True
            lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_pitch"]
            if (wrist_pitch < lower_limit) or (wrist_pitch > upper_limit):
                limits_violated = True
            lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_roll"]
            if (wrist_roll < lower_limit) or (wrist_roll > upper_limit):
                limits_violated = True

            ################################################################
            # DROP GRIPPER ORIENTATION GOALS WITH LARGE JOINT ANGLE CHANGES
            #
            # Dropping goals that result in extreme changes in joint
            # angles over a single time step avoids the nearly 360
            # degree rotation in an opposite direction of motion that
            # can occur when a goal jumps across a joint limit for a
            # joint with a large range of motion like the roll joint.
            #
            # This also reduces the potential for unexpected wrist
            # motions near gimbal lock when the yaw and roll axes are
            # aligned (i.e., the gripper is pointed down to the
            # ground). Goals representing slow motions that traverse
            # near this gimbal lock region can still result in the
            # gripper approximately going upside down in a manner
            # similar to a pendulum, but this results in large yaw
            # joint motions and is prevented at high speeds due to
            # joint angles that differ significantly between time
            # steps. Inverting this motion must also be performed at
            # low speeds or the gripper will become stuck and need to
            # traverse a trajectory around the gimbal lock region.
            #
            extreme_difference_violated = False
            if self.drop_extreme_wrist_orientation_change:
                prev_wrist_yaw = self.prev_commanded_wrist_orientation["joint_wrist_yaw"]
                if prev_wrist_yaw is not None:
                    diff = abs(wrist_yaw - prev_wrist_yaw)
                    if diff > self.max_allowed_wrist_yaw_change:
                        print(
                            "extreme wrist_yaw change of {:.2f} deg".format(
                                (180.0 * (diff / np.pi))
                            )
                        )
                        extreme_difference_violated = True
                prev_wrist_roll = self.prev_commanded_wrist_orientation["joint_wrist_roll"]
                if prev_wrist_roll is not None:
                    diff = abs(wrist_roll - prev_wrist_roll)
                    if diff > self.max_allowed_wrist_roll_change:
                        print(
                            "extreme wrist_roll change of {:.2f} deg".format(
                                (180.0 * (diff / np.pi))
                            )
                        )
                        extreme_difference_violated = True
            #
            ################################################################

            if self.debug_wrist_orientation:
                if limits_violated:
                    print("The wrist angle limits were violated.")

            if (not extreme_difference_violated) and (not limits_violated):
                new_wrist_orientation = np.array([wrist_yaw, wrist_pitch, wrist_roll])

                # Use exponential smoothing to filter the wrist
                # orientation configuration used to command the
                # robot.
                self.filtered_wrist_orientation = (
                    (1.0 - self.wrist_orientation_filter) * self.filtered_wrist_orientation
                ) + (self.wrist_orientation_filter * new_wrist_orientation)

                new_goal_configuration["joint_wrist_yaw"] = self.filtered_wrist_orientation[0]
                new_goal_configuration["joint_wrist_pitch"] = self.filtered_wrist_orientation[1]
                new_goal_configuration["joint_wrist_roll"] = self.filtered_wrist_orientation[2]

                self.prev_commanded_wrist_orientation = {
                    "joint_wrist_yaw": self.filtered_wrist_orientation[0],
                    "joint_wrist_pitch": self.filtered_wrist_orientation[1],
                    "joint_wrist_roll": self.filtered_wrist_orientation[2],
                }

            # Convert from the absolute goal for the mobile
            # base to an incremental move to be performed
            # using rotate_by. This should be performed just
            # before sending the commands to make sure it's
            # using the most rececnt mobile base angle
            # estimate to reduce overshoot and other issues.

            # convert base odometry angle to be in the range -pi to pi
            # negative is to the robot's right side (counterclockwise)
            # positive is to the robot's left side (clockwise)

            # TODO
            # Following line used by simple_IK, but can only be run on stretch due to need for robot status.
            # base_odom_theta = hm.angle_diff_rad(self.robot.base.status["theta"], 0.0)
            # current_mobile_base_angle = base_odom_theta

            # Compute base rotation to reach the position determined by the IK solver
            # Else we will just use the computed value from IK for now to see if that works
            if self._use_simple_gripper_rules:
                new_goal_configuration[
                    "joint_mobile_base_rotation"
                ] = self.filtered_wrist_position_configuration[0]
            else:
                # Clear this, let us rotate freely
                current_mobile_base_angle = 0
                # new_goal_configuration[
                #    "joint_mobile_base_rotation"
                # ] += self.filtered_wrist_position_configuration[0]

            # Figure out how much we are allowed to rotate left or right
            # new_goal_configuration["joint_mobile_base_rotate_by"] = np.clip(
            #     new_goal_configuration["joint_mobile_base_rotation"] - current_mobile_base_angle,
            #     -self.max_rotation_change,
            #     self.max_rotation_change,
            # )
            if self.teleop_mode == "base_x":
                new_goal_configuration["joint_mobile_base_rotate_by"] = 0.0

                # Base_x_origin is reset to base_x coordinate at start of demonstration
                if self.base_x_origin is None:
                    self.base_x_origin = current_state["base_x"]

                self.current_base_x = current_state["base_x"] - self.base_x_origin

                new_goal_configuration["joint_mobile_base_translate_by"] = (
                    new_goal_configuration["joint_mobile_base_translation"] - self.current_base_x
                )

            if self.debug_base_rotation:
                print()
                print("Debugging base rotation:")
                print(f"{new_goal_configuration['joint_mobile_base_rotation']=}")
                print(f"{self.filtered_wrist_position_configuration[0]=}")
                print(f"{new_goal_configuration['joint_mobile_base_rotate_by']=}")
                print("ROTATE BY:", new_goal_configuration["joint_mobile_base_rotate_by"])

            # remove virtual joint and approximate motion with rotate_by using joint_mobile_base_rotate_by
            # del new_goal_configuration["joint_mobile_base_rotation"]

        return new_goal_configuration

    def __del__(self):
        self.goal_send_socket.close()
        if self._recording or self._need_to_write:
            self._recorder.write()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="")
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
    parser.add_argument("-P", "--send_port", type=int, default=4406, help="Port to send goals to.")
    parser.add_argument(
        "-R",
        "--replay_filename",
        type=str,
        default=None,
        help="The filename of the recorded session to replay, if set..",
    )
    parser.add_argument("--display_point_cloud", action="store_true")
    parser.add_argument(
        "--teleop-mode",
        "--teleop_mode",
        type=str,
        default="base_x",
        choices=["stationary_base", "base_x"],
    )
    parser.add_argument("--record-success", action="store_true", help="Record success of episode.")
    parser.add_argument("--show-aruco", action="store_true", help="Show aruco debug information.")
    parser.add_argument("--platform", type=str, default="linux", choices=["linux", "not_linux"])
    args = parser.parse_args()

    client = RobotClient(
        use_remote_computer=True,
        robot_ip=args.robot_ip,
        d405_port=args.d405_port,
        verbose=args.verbose,
    )

    # Create dex teleop leader - this will detect markers and send off goal dicts to the robot.
    evaluator = DexTeleopLeader(
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
        force_record=args.force,
        display_point_cloud=args.display_point_cloud,
        save_images=args.save_images,
        send_port=args.send_port,
        robot_ip=args.robot_ip,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        debug_aruco=args.show_aruco,
        platform=args.platform,
    )
    try:
        client.run(evaluator)
    except KeyboardInterrupt:
        pass
