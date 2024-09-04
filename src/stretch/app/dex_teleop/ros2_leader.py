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

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.app.dex_teleop.goal_from_teleop as gt
import stretch.app.dex_teleop.webcam_teleop_interface as wt
import stretch.motion.constants as constants
import stretch.motion.simple_ik as si
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient

try:
    from stretch.app.dex_teleop.hand_tracker import HandTracker
except ImportError as e:
    print("Hand tracker not available. Please install its dependencies if you want to use it.")
    print()
    print("\tpython -m pip install .[hand_tracker]")
    print()
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder


class ZmqRos2Leader:
    """Leader class for DexTeleop using the Zmq_client for ROS2 on Stretch"""

    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        left_handed: bool = False,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        force_record: bool = False,
        debug_aruco: bool = False,
        save_images: bool = False,
        teleop_mode: str = "base_x",
        record_success: bool = False,
        platform: str = "linux",
        use_clutch: bool = False,
        teach_grasping: bool = False,
    ):
        self.robot = robot
        self.camera = None

        # TODO: fix these two things
        manipulate_on_ground = False
        slide_lift_range = False
        self.save_images = save_images
        self.teleop_mode = teleop_mode
        self.record_success = record_success
        self.platform = platform
        self.verbose = verbose
        self.use_clutch = use_clutch
        self.teach_grasping = teach_grasping

        self.left_handed = left_handed

        self.base_x_origin = None
        self.current_base_x = 0.0

        self.use_gripper_center = True

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
            "backend": "ros2",
        }

        self._force = force_record
        self._recording = False or self._force
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            data_dir, task_name, user_name, env_name, save_images, self.metadata, fps=6
        )
        self.prev_goal_dict = None

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
        relative: bool = False,
        verbose: bool = False,
        **config,
    ):

        res, success, info = self.robot._robot_model.manip_ik_solver.compute_ik(
            wrist_position,
            gripper_orientation,
            ignore_missing_joints=True,
        )
        new_goal_configuration = self.robot._robot_model.manip_ik_solver.q_array_to_dict(res)
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
                        new_goal_configuration["base_x_joint"],
                        new_goal_configuration["joint_lift"],
                        new_goal_configuration["joint_arm_l0"],
                    ]
                )
            else:
                new_wrist_position_configuration = np.array(
                    [
                        new_goal_configuration["base_theta_joint"],
                        new_goal_configuration["joint_lift"],
                        new_goal_configuration["joint_arm_l0"],
                    ]
                )

            # Arm scaling
            new_wrist_position_configuration[2] = (
                new_wrist_position_configuration[2] * dt.ros2_arm_scaling_factor
            )

            # Use exponential smoothing to filter the wrist
            # position configuration used to command the
            # robot.
            self.filtered_wrist_position_configuration = (
                (1.0 - self.wrist_position_filter) * self.filtered_wrist_position_configuration
            ) + (self.wrist_position_filter * new_wrist_position_configuration)

            new_goal_configuration["joint_lift"] = self.filtered_wrist_position_configuration[1]
            new_goal_configuration["joint_arm_l0"] = self.filtered_wrist_position_configuration[2]

            #################################

            #################################
            # INPUT: grip_width between 0.0 and 1.0

            if (grip_width is not None) and (grip_width > -1000.0):
                # Use width to interpolate between open and closed
                new_goal_configuration["stretch_gripper"] = (
                    self.robot._robot_model.GRIPPER_CLOSED
                ) + grip_width * (
                    abs(self.robot._robot_model.GRIPPER_OPEN)
                    + abs(self.robot._robot_model.GRIPPER_CLOSED)
                )

            ##################################################
            # INPUT: x_axis, y_axis, z_axis

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

            # Teleop mode specific modifications
            # if self.teleop_mode == "base_x":
            #     new_goal_configuration["joint_mobile_base_rotate_by"] = 0.0

            #     # Base_x_origin is reset to base_x coordinate at start of demonstration
            #     if self.base_x_origin is None:
            #         self.base_x_origin = current_state["base_x"]

            #     self.current_base_x = current_state["base_x"] - self.base_x_origin

            #     new_goal_configuration["joint_mobile_base_translate_by"] = (
            #         new_goal_configuration["joint_mobile_base_translation"] - self.current_base_x
            #     )

        return new_goal_configuration

    def run(self, display_received_images):
        loop_timer = lt.LoopStats("dex_teleop_leader")

        if self.use_clutch:
            hand_tracker = HandTracker(left_clutch=(not self.left_handed))

        print("=== Starting Dex Teleop Leader ===")
        print("Press spacebar to start/stop recording.")
        if self.teach_grasping:
            print("Press 1, 2, or 3 to teach PREGRASP, GRASP, or POSTGRASP.")
        print("Press 0-9 to record waypoints.")
        if self.record_success:
            print("Press y/n to record success/failure of episode after each episode.")
        if self.use_clutch:
            print("Clutch mode enabled. Place an empty hand over the webcam to clutch.")
        print("Press ESC to exit.")

        # loop stuff for clutch
        clutched = False
        clutch_debounce_threshold = 3
        change_clutch_count = 0
        check_hand_frame_skip = 3
        i = 0
        max_i = 100  # arbitrary number of iterations

        last_robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        offset_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            while True:
                waypoint_key = None

                loop_timer.mark_start()

                # Get observation
                observation = self.robot.get_servo_observation()

                # Process images
                gripper_color_image = cv2.cvtColor(observation.ee_rgb, cv2.COLOR_RGB2BGR)
                gripper_depth_image = (
                    observation.ee_depth.astype(np.float32) * observation.ee_depth_scaling
                )

                head_color_image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
                head_depth_image = observation.depth.astype(np.float32) * observation.depth_scaling

                if display_received_images:
                    # change depth to be h x w x 3
                    depth_image_x3 = np.stack((gripper_depth_image,) * 3, axis=-1)
                    combined = np.hstack((gripper_color_image / 255, depth_image_x3 / 4))

                    # Head images
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
                elif key == 27:
                    if self._recording:
                        self._need_to_write = True
                    self._recording = False
                    print("[LEADER] Recording stopped. Terminating.")
                    break
                else:
                    for i in range(10):
                        if key == ord(str(i)):
                            if self.teach_grasping and i >= 1 and i <= 3:
                                if i == 1:
                                    print(f"[LEADER] Key {i} pressed. Teaching PREGRASP.")
                                elif i == 2:
                                    print(f"[LEADER] Key {i} pressed. Teaching GRASP.")
                                elif i == 3:
                                    print(f"[LEADER] Key {i} pressed. Teaching POSTGRASP.")
                            else:
                                print(f"[LEADER] Key {i} pressed. Recording waypoint {i}.")
                            waypoint_key = i
                            break

                # Raw input from teleop
                markers, color_image = self.webcam_aruco_detector.process_next_frame()

                # Set up commands to be sent to the robot
                goal_dict = self.goal_from_markers.get_goal_dict(markers)

                if self.use_clutch:
                    # check if n-th frame - if so, check clutch
                    if i % check_hand_frame_skip == 0:
                        hand_prediction = hand_tracker.run_detection(color_image)
                        check_clutched = hand_tracker.check_clutched(hand_prediction)

                        # debounce
                        if check_clutched != clutched:
                            change_clutch_count += 1
                        else:
                            change_clutch_count = 0

                        if change_clutch_count >= clutch_debounce_threshold:
                            clutched = not clutched
                            change_clutch_count = 0

                    i += 1
                    i = i % max_i

                if goal_dict is not None:
                    # Convert goal dict into a quaternion
                    goal_dict = dt_utils.process_goal_dict(
                        goal_dict, self.prev_goal_dict, self.use_gripper_center
                    )
                else:
                    # Goal dict that is not worth processing
                    goal_dict = {"valid": False}

                if goal_dict["valid"]:
                    # get robot configuration
                    goal_configuration = self.get_goal_joint_config(**goal_dict)

                    # Format to standard action space
                    goal_configuration = dt_utils.format_actions(goal_configuration)

                    if self._recording:
                        print("[LEADER] goal_dict =")
                        pp.pprint(goal_configuration)

                    robot_pose = np.array(
                        [
                            goal_configuration["base_x_joint"],
                            goal_configuration["joint_lift"],
                            goal_configuration["joint_arm_l0"],
                            goal_configuration["joint_wrist_yaw"],
                            goal_configuration["joint_wrist_pitch"],
                            goal_configuration["joint_wrist_roll"],
                        ]
                    )

                    if not clutched:
                        last_robot_pose = robot_pose

                        # add clutch offset
                        robot_pose += offset_pose

                        self.robot.arm_to(
                            robot_pose,
                            gripper=goal_configuration["stretch_gripper"],
                            head=constants.look_at_ee,
                        )

                        # Prep joint states as dict
                        joint_states = {
                            k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
                        }
                        if self._recording and self.prev_goal_dict is not None:
                            self._recorder.add(
                                ee_rgb=gripper_color_image,
                                ee_depth=gripper_depth_image,
                                xyz=goal_dict["relative_gripper_position"],
                                quaternion=goal_dict["relative_gripper_orientation"],
                                gripper=goal_dict["grip_width"],
                                head_rgb=head_color_image,
                                head_depth=head_depth_image,
                                observations=joint_states,
                                actions=goal_configuration,
                                ee_pos=observation.ee_camera_pose,
                                ee_rot=observation.ee_camera_pose,
                            )

                            # Record waypoint
                            if waypoint_key is not None:
                                print("[LEADER] Recording waypoint.")
                                ok = self._recorder.add_waypoint(
                                    waypoint_key,
                                    robot_pose,
                                    goal_configuration["stretch_gripper"],
                                )
                                if not ok:
                                    logger.warning(
                                        f"[LEADER] WARNING: overwriting previous waypoint {waypoint_key}."
                                    )
                    else:
                        offset_pose = last_robot_pose - robot_pose
                elif waypoint_key is not None:
                    print("[LEADER] Recording waypoint failed. Commanded goal_dict was invalid.")

                self.prev_goal_dict = goal_dict

                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()

                if self._need_to_write:
                    if self.record_success:
                        success = self.ask_for_success()
                        print("[LEADER] Writing data to disk with success = ", success)
                        self._recorder.write(success=success)
                    else:
                        print("[LEADER] Writing data to disk.")
                        self._recorder.write()
                    self._need_to_write = False

        finally:
            print("Exiting...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--user-name", type=str, default="default_user")
    parser.add_argument("-t", "--task-name", type=str, default="default_task")
    parser.add_argument("-e", "--env-name", type=str, default="default_env")
    parser.add_argument("-f", "--force", action="store_true", help="Force data recording.")
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
    parser.add_argument("--show-aruco", action="store_true", help="Show aruco debug information.")
    parser.add_argument("--platform", type=str, default="linux", choices=["linux", "not_linux"])
    parser.add_argument("-c", "--clutch", action="store_true")
    parser.add_argument("--teach-grasping", action="store_true")
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
        enable_rerun_server=False,
    )
    robot.switch_to_manipulation_mode()
    robot.move_to_manip_posture()

    leader = ZmqRos2Leader(
        robot=robot,
        verbose=args.verbose,
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
        force_record=args.force,
        save_images=args.save_images,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        platform=args.platform,
        use_clutch=args.clutch,
        teach_grasping=args.teach_grasping,
    )

    try:
        leader.run(display_received_images=True)
    except KeyboardInterrupt:
        pass

    robot.stop()
