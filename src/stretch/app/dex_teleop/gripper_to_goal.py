# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import errno
import math
import os
import pprint as pp
import time
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import stretch_body.robot as rb
import urchin as urdf_loader
from scipy.spatial.transform import Rotation

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import stretch.app.dex_teleop.robot_move as rm
import stretch.motion.simple_ik as si
import stretch.utils.loop_stats as lt

# This tells us if we are using the gripper center for control or not
from stretch.app.dex_teleop.leader import use_gripper_center
from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver
from stretch.utils.geometry import angle_difference, get_rotation_from_xyz


def load_urdf(file_name):
    if not os.path.isfile(file_name):
        print()
        print("*****************************")
        print(
            "ERROR: "
            + file_name
            + " was not found. Simple IK requires a specialized URDF saved with this file name. prepare_base_rotation_ik_urdf.py can be used to generate this specialized URDF."
        )
        print("*****************************")
        print()
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    urdf = urdf_loader.URDF.load(file_name, lazy_load_meshes=True)
    return urdf


def nan_in_configuration(configuration):
    for k, v in configuration.items():
        if math.isnan(v) or np.isnan(v):
            return True
    return False


class GripperToGoal:

    _ee_link_name = "link_grasp_center"

    # This is set to enable us to flip back to simple IK
    _use_simple_gripper_rules = not use_gripper_center
    max_rotation_change = 0.5

    debug_base_rotation: bool = False

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
        starting_configuration,
        robot: rb.Robot,
        robot_move: rm.RobotMove,
        simple_ik: Optional[si.SimpleIK] = None,
        robot_allowed_to_move: bool = True,
        using_stretch_2: bool = False,
    ):
        if using_stretch_2:
            self.grip_range = dt.dex_wrist_grip_range
        else:
            self.grip_range = dt.dex_wrist_3_grip_range

        self.using_stretch_2 = using_stretch_2
        self.robot_move = robot_move
        self.robot = robot

        self.joints_allowed_to_move = [
            "stretch_gripper",
            "joint_arm_l0",
            "joint_lift",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            # "joint_mobile_base_rotate_by",
            "joint_mobile_base_translate_by",
        ]
        self._ik_joints_allowed_to_move = [
            "joint_arm_l0",
            "joint_lift",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            # "joint_mobile_base_rotation",
            "joint_mobile_base_translation",
        ]

        # Get Wrist URDF joint limits
        # rotary_urdf_file_name = "./stretch_base_rotation_ik_with_fixed_wrist.urdf"
        # rotary_urdf = load_urdf(rotary_urdf_file_name)
        translation_urdf_file_name = "./stretch_base_translation_ik_with_fixed_wrist.urdf"
        translation_urdf = load_urdf(translation_urdf_file_name)
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

        self.robot_allowed_to_move = robot_allowed_to_move
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

        if self._use_simple_gripper_rules:
            # Initialize IK
            if self.simple_ik is None:
                print("Simple IK setup")
                self.simple_ik = si.SimpleIK()
                print("Done")
        else:
            self.simple_ik = None

        self.print_robot_status_thread_timing = False
        self.debug_wrist_orientation = False

        self.max_allowed_wrist_yaw_change = dt.max_allowed_wrist_yaw_change
        self.max_allowed_wrist_roll_change = dt.max_allowed_wrist_roll_change

    def get_current_config(self):
        state = {
            "base_x": self.robot.status["base"]["x"],
            "base_x_vel": self.robot.status["base"]["x_vel"],
            "base_y": self.robot.status["base"]["y"],
            "base_y_vel": self.robot.status["base"]["y_vel"],
            "base_theta": self.robot.status["base"]["theta"],
            "base_theta_vel": self.robot.status["base"]["theta_vel"],
            "joint_lift": self.robot.status["lift"]["pos"],
            "joint_arm_l0": self.robot.status["arm"]["pos"],
            "joint_wrist_pitch": self.robot.status["end_of_arm"]["wrist_pitch"]["pos"],
            "joint_wrist_yaw": self.robot.status["end_of_arm"]["wrist_yaw"]["pos"],
            "joint_wrist_roll": self.robot.status["end_of_arm"]["wrist_roll"]["pos"],
            "stretch_gripper": self.robot.status["end_of_arm"]["stretch_gripper"]["pos_pct"],
        }
        return state

    def get_current_ee_pose(self):
        state = self.get_current_config()
        return self.manip_ik_solver.compute_fk(state, ignore_missing_joints=True)

    def __del__(self):
        if self.robot is not None:
            print("GripperToGoal.__del__: stopping the robot")
            self.robot.stop()

    def get_wrist_position(self, rotation: Rotation):
        """Get wrist position with some safety and lower limits to makes sure it's safe to execute.

        Args:
            rotation(Rotation): computed goal rotation of the wrist, unsafe

        Returns:
            wrist_yaw, wrist_pitch, wrist_roll: safe, executable positions
        """
        # capital letters represent intrinsic
        # rotations, lowercase letters represent
        # extrinsic rotations
        ypr = rotation.as_euler("ZXY", degrees=False)

        wrist_yaw = angle_difference(ypr[0] + np.pi, 0.0)
        lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_yaw"]
        if wrist_yaw < lower_limit:
            wrist_yaw = wrist_yaw + (2.0 * np.pi)

        wrist_pitch = ypr[1]

        wrist_roll = angle_difference(ypr[2] + np.pi, 0.0)
        return wrist_yaw, wrist_pitch, wrist_roll

    def update_goal(
        self,
        grip_width,
        wrist_position: np.ndarray,
        gripper_orientation: np.ndarray,
        relative=False,
        **config,
    ):

        ###############################
        # INPUT: wrist_position
        if "use_gripper_center" in config and config["use_gripper_center"] != use_gripper_center:
            raise RuntimeError("leader and follower are not set up to use the same target poses.")

        # Use Simple IK to find configurations for the
        # mobile base angle, lift distance, and arm
        # distance to achieve the goal wrist position in
        # the world frame.
        if self._use_simple_gripper_rules:
            new_goal_configuration = self.simple_ik.ik_rotary_base(wrist_position)
        else:
            res, success, info = self.manip_ik_solver.compute_ik(
                wrist_position, gripper_orientation, q_init=self.get_current_config()
            )
            new_goal_configuration = self.manip_ik_solver.q_array_to_dict(res)
            if not success:
                print("!!! BAD IK SOLUTION !!!")
                new_goal_configuration = None
            pp.pp(new_goal_configuration)

        if new_goal_configuration is None:
            print(
                f"WARNING: IK failed to find a valid new_goal_configuration so skipping this iteration by continuing the loop. Input to IK: wrist_position = {wrist_position}, Output from IK: new_goal_configuration = {new_goal_configuration}"
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
            base_odom_theta = angle_difference(self.robot.base.status["theta"], 0.0)
            current_mobile_base_angle = base_odom_theta

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
            new_goal_configuration["joint_mobile_base_rotate_by"] = np.clip(
                new_goal_configuration["joint_mobile_base_rotation"] - current_mobile_base_angle,
                -self.max_rotation_change,
                self.max_rotation_change,
            )
            if self.debug_base_rotation:
                print()
                print("Debugging base rotation:")
                print(f"{new_goal_configuration['joint_mobile_base_rotation']=}")
                print(f"{self.filtered_wrist_position_configuration[0]=}")
                print(f"{new_goal_configuration['joint_mobile_base_rotate_by']=}")
            print("ROTATE BY:", new_goal_configuration["joint_mobile_base_rotate_by"])
            # remove virtual joint and approximate motion with rotate_by using joint_mobile_base_rotate_by
            del new_goal_configuration["joint_mobile_base_rotation"]

            # If motion allowed, command the robot to move to the target configuration
            if self.robot_allowed_to_move:
                if nan_in_configuration(new_goal_configuration):
                    print()
                    print("******************************************************************")
                    print(
                        "WARNING: dex_teleop: new_goal_configuration has a nan, so skipping execution on the robot"
                    )
                    print()
                    print("     new_goal_configuration =", new_goal_configuration)
                    print()
                    print("******************************************************************")
                    print()
                else:
                    self.robot_move.to_configuration(
                        new_goal_configuration, self.joints_allowed_to_move
                    )
                    self.robot.push_command()

            # Print robot status timing stats, if desired.
            if self.print_robot_status_thread_timing:
                self.robot.non_dxl_thread.stats.pretty_print()
                print()
                self.robot.dxl_end_of_arm_thread.stats.pretty_print()
                print()
                self.robot.dxl_head_thread.stats.pretty_print()
                print()
                self.robot.sys_thread.stats.pretty_print()

    def execute_goal(self, new_goal_configuration):
        # If motion allowed, command the robot to move to the target configuration
        if self.robot_allowed_to_move:
            if nan_in_configuration(new_goal_configuration):
                print()
                print("******************************************************************")
                print(
                    "WARNING: dex_teleop: new_goal_configuration has a nan, so skipping execution on the robot"
                )
                print()
                print("     new_goal_configuration =", new_goal_configuration)
                print()
                print("******************************************************************")
                print()
            else:
                self.robot_move.to_configuration(
                    new_goal_configuration, self.joints_allowed_to_move
                )
                self.robot.pimu.set_fan_on()
                self.robot.push_command()

        # Print robot status timing stats, if desired.
        if self.print_robot_status_thread_timing:
            self.robot.non_dxl_thread.stats.pretty_print()
            print()
            self.robot.dxl_end_of_arm_thread.stats.pretty_print()
            print()
            self.robot.dxl_head_thread.stats.pretty_print()
            print()
            self.robot.sys_thread.stats.pretty_print()


if __name__ == "__main__":

    args = dt.get_arg_parser().parse_args()
    use_fastest_mode = args.fast
    manipulate_on_ground = args.ground
    left_handed = args.left
    using_stretch_2 = args.stretch_2
    use_multiprocessing = args.multiprocessing

    # When False, the robot should only move to its initial position
    # and not move in response to ArUco markers. This is helpful when
    # first trying new code and interface objects.
    robot_allowed_to_move = True

    lift_middle = dt.get_lift_middle(manipulate_on_ground)
    center_configuration = dt.get_center_configuration(lift_middle)
    starting_configuration = dt.get_starting_configuration(lift_middle)

    gripper_to_goal = GripperToGoal(starting_configuration, robot_allowed_to_move, using_stretch_2)

    if use_multiprocessing:
        shm = shared_memory.SharedMemory(name=dt.shared_memory_name, create=False)
        example_goal_array = dt.get_example_goal_array()
        received_goal_array = np.ndarray(
            example_goal_array.shape, dtype=example_goal_array.dtype, buffer=shm.buf
        )
    else:

        shm = None
        goal_grip_width = 1.0
        if not use_gripper_center:
            # position': array([ 0.07624164, -0.42347259,  0.84443279]),
            goal_wrist_position = np.array([-0.03, -0.4, 0.9])
            goal_x_axis = np.array([1.0, 0.0, 0.0])
            goal_y_axis = np.array([0.0, -1.0, 0.0])
            goal_z_axis = np.array([0.0, 0.0, -1.0])
        else:
            goal_wrist_position = np.array([-0.03, -0.6, 0.9])
            goal_x_axis = np.array([0.0, -1.0, 0.0])
            goal_y_axis = np.array([1.0, 0.0, 0.0])
            goal_z_axis = np.array([0.0, 0.0, 1.0])
        goal_dict = {
            "grip_width": goal_grip_width,
            "wrist_position": goal_wrist_position,
            "gripper_x_axis": goal_x_axis,
            "gripper_y_axis": goal_y_axis,
            "gripper_z_axis": goal_z_axis,
        }

    # Compute quaternion for end effector position
    r = get_rotation_from_xyz(
        goal_dict["gripper_x_axis"],
        goal_dict["gripper_y_axis"],
        goal_dict["gripper_z_axis"],
    )
    goal_dict["gripper_orientation"] = r.as_quat()

    print("Ready to go")
    loop_timer = lt.LoopTimer()
    print_timing = False
    first_goal_received = False

    try:
        while True:
            loop_timer.mark_start()

            print("Commanded goal =")
            pp.pp(goal_dict)
            print("Current position =")
            pp.pp(gripper_to_goal.get_current_config())
            print("Current ee pose =")
            pos, quat = gripper_to_goal.get_current_ee_pose()
            r = Rotation.from_quat(quat)
            T = r.as_matrix()
            ee_x = T[:3, 0]
            ee_y = T[:3, 1]
            ee_z = T[:3, 2]
            pp.pp(
                {
                    "position": pos,
                    "orientation": quat,
                    "ee_x_axis": ee_x,
                    "ee_y_axis": ee_y,
                    "ee_z_axis": ee_z,
                }
            )

            if use_multiprocessing:
                if not dt.is_a_do_nothing_goal_array(received_goal_array):
                    goal_dict = dt.goal_array_to_dict(received_goal_array)
                    gripper_to_goal.update_goal(**goal_dict)
                    print("goal_dict =")
                    pp.pprint(goal_dict)
                    if not first_goal_received:
                        loop_timer.reset()
                        first_goal_received = True
                else:
                    print("received do nothing goal array")
            else:
                gripper_to_goal.update_goal(**goal_dict)
                if not first_goal_received:
                    loop_timer.reset()
                    first_goal_received = True

            time.sleep(0.03)
            loop_timer.mark_end()
            if print_timing:
                loop_timer.pretty_print()
    finally:
        if shm is not None:
            print("cleaning up shared memory multiprocessing")
            shm.close()
