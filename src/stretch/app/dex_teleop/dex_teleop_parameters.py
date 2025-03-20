# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import argparse

import numpy as np


def get_arg_parser():
    parser = argparse.ArgumentParser(
        prog="Stretch Dexterous Teleop",
        description="This application enables a human operator to control Stretch using ArUco markers and a Logitech C930e webcam on the floor looking up with a light ring. Currently, the code supports tongs with ArUco markers attached.",
    )
    parser.add_argument(
        "-l",
        "--left",
        action="store_true",
        help="Use Stretch as a left-handed robot. This requires left-hand tongs, which have distinct ArUco markers from right-hand tongs.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Move Stretch at the fastest available speed. The default is to move the robot at the slowest available speed",
    )
    parser.add_argument(
        "-g",
        "--ground",
        action="store_true",
        help="Manipulate at ground level up to around tabletop height. The default is to manipulate from tabletop height up to countertop height.",
    )
    parser.add_argument(
        "--stretch_2",
        action="store_true",
        help="Use a Stretch 2, which may require special settings.",
    )
    parser.add_argument(
        "-m",
        "--multiprocessing",
        action="store_true",
        help="Write goals to shared memory using Python multiprocessing.",
    )
    parser.add_argument(
        "-i",
        "--slide_lift_range",
        action="store_true",
        help="Holding the tongs high will gradually slide the lift range of motion upward. Holding them low will gradually slide the lift range of motion downward. The default is to use a fixed range of motion for the lift.",
    )
    parser.add_argument(
        "-p",
        "--send-port",
        type=int,
        default=4405,
        help="Set the port used for sending d405 images.",
    )
    parser.add_argument(
        "-r",
        "--recv-port",
        type=int,
        default=4406,
        help="Set the port used for receiving actions.",
    )
    parser.add_argument(
        "--gamma",
        action="store",
        type=float,
        default=2.0,
        help="Set the gamma correction factor for the images.",
    )
    parser.add_argument(
        "-e",
        "--exposure",
        action="store",
        type=str,
        default="low",
        help="Set the D405 exposure to {dh.exposure_keywords} or an integer in the range {dh.exposure_range}",
    )
    parser.add_argument(
        "--scaling",
        action="store",
        type=float,
        default=0.5,
        help="Set the scaling factor for the images.",
    )
    parser.add_argument(
        "--leader-ip",
        action="store",
        type=str,
        default="192.168.1.169",
        help="Set the IP of dex teleop leader when running with remote computer",
    )

    return parser


# Measurements for the tongs
tongs_to_use = "56mm"  # '50mm' #'44mm'

if tongs_to_use == "56mm":
    # 50mm ArUco marker tongs
    tongs_cube_side = 0.0735
    tongs_pin_joint_to_marker_center = 0.112
    tongs_pin_joint_to_tong_tip = 0.102
    # tongs_marker_center_to_tong_tip = (tongs_cube_side / 2.0) + 0.0125
    tongs_marker_center_to_tong_tip = (tongs_cube_side / 2.0) + 0.0105
    tongs_open_grip_width = 0.145
    tongs_closed_grip_width = 0.088


# The maximum and minimum goal_wrist_position z values do not
# need to be perfect due to joint limit checking performed by
# the SimpleIK based on the specialized URDF joint
# limits. They are specified with respect to the robot's
# coordinate system.
goal_max_position_z = 1.35
goal_min_position_z = 0.07


# Regions at the top and bottom of the allowable tongs range
# are reserved for changing the range over which the lift is
# operating. This sliding region enables a user to use the
# lift's full range without restarting the code.
lift_sliding_region_height = 0.5


# Set how fast the lift will be translated when being slid.
lift_range_sliding_speed_multiplier = 10.0  # 4.0
lift_range_offset_change_per_timestep = 0.001 * lift_range_sliding_speed_multiplier


dex_wrist_3_grip_range = 400.0
dex_wrist_grip_range = 200.0


# Lower limit for wrist pitch. Allowing angles close to -Pi/2 will
# allow the gripper to point almost straight down, which is near a
# singularity. When the wrist is straight down with pitch = -Pi/2, the
# wrist roll and wrist yaw joints have rotational axes that are
# parallel resulting in gimbal lock. This can result in undesirable
# motions and unintuitive configurations. Changing this variable can
# be used to increase the risk of issues in order to point the gripper
# closer to straight down. A value of None will result in the pitch
# being allowed to move over the entire range of motion allowed by the
# robot's exported URDF. The default lowest pitch was -1.57 rad on the
# default URDF used during development (-89.954373836 deg)

# AFTER CHANGING THIS VARIABLE YOU NEED TO RUN
# prepare_specialized_urdfs.py FOR IT TO TAKE EFFECT.

wrist_pitch_lower_limit = -0.8 * (np.pi / 2.0)


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
# max_allowed_wrist_yaw_change = np.pi/2.0
# max_allowed_wrist_roll_change = np.pi/2.0
max_allowed_wrist_yaw_change = 1.8 * (np.pi / 2.0)
max_allowed_wrist_roll_change = 1.8 * (np.pi / 2.0)


# This is the weight between 0.0 and 1.0 multiplied by the current
# command when performing exponential smoothing. A higher weight leads
# to less lag, but higher noise. A lower weight leads to more lag, but
# smoother motions with less noise.

# current_filtered_value = ((1.0 - exponential_smoothing) * previous_filtered_value) + (exponential_smoothing * current_value)
exponential_smoothing_for_orientation = 0.2  # 0.25 #0.1
exponential_smoothing_for_position = 0.2  # 0.5 #0.1


# When False, the robot should only move to its initial position
# and not move in response to ArUco markers. This is helpful when
# first trying new code and interface objects.
robot_allowed_to_move = True

# Camera stand at maximal height
# Minimum distance from the tongs to the camera in meters
# min_dist_from_camera_to_tongs = 0.3
# Maximum distance from the tongs to the camera in meters
# max_dist_from_camera_to_tongs = 0.8

# Camera stand at minimal height

# This range represents the manipulation range. Commands for sliding
# the lift range are outside of this range.

# Minimum distance from the tongs to the camera in meters
# min_dist_from_camera_to_tongs = 0.6  # 0.5
# Maximum distance from the tongs to the camera in meters
# max_dist_from_camera_to_tongs = 1.0  # 1.0

# Custom range for ACT demonstrations
# Minimum distance from the tongs to the camera in meters
min_dist_from_camera_to_tongs = 0.8
# Maximum distance from the tongs to the camera in meters
max_dist_from_camera_to_tongs = 1.4

# Maximum height range of tongs
max_tongs_height_range = max_dist_from_camera_to_tongs - min_dist_from_camera_to_tongs


# The origin for teleoperation with respect to the camera's frame
# of reference. Holding the teleoperation interface at this
# position with respect to the camera results in the robot being
# commanded to achieve the center_wrist_position. The camera's
# frame of reference has its origin in the optical center of the
# camera and its z-axis points directly out of the camera on the
# camera's optical axis.
teleop_origin_x = 0.0
teleop_origin_z = min_dist_from_camera_to_tongs + max_tongs_height_range / 2.0
teleop_origin_y = 0.24

teleop_origin = np.array([teleop_origin_x, teleop_origin_y, teleop_origin_z])

# Supported modes of teleoperation, used to define state and action formatting
SUPPORTED_MODES = ["standard", "rotary_base", "stationary_base", "base_x", "old_stationary_base"]

# Scaling factor for arm from IK to ros2 backend
ros2_arm_scaling_factor = 3.8

# Dex teleop controlled joints
DEX_TELEOP_CONTROLLED_JOINTS = [
    "base_x_joint",
    "base_y_joint",
    "base_theta_joint",
    "joint_arm_l0",
    "joint_lift",
    "joint_wrist_roll",
    "joint_wrist_pitch",
    "joint_wrist_yaw",
    "stretch_gripper",
]
# Robot configuration used to define the center wrist position


def get_lift_middle(manipulate_on_ground):
    # Additional joint limits for the lift beyond what the URDF
    # specifies. A None value is ignored in favor of the URDF joint
    # limits. These can only be more restrictive than the URDF joint
    # limits.
    if manipulate_on_ground:
        # lift_maximum = max_tongs_height_range
        lift_middle = max_tongs_height_range / 2.0
    else:
        lift_minimum = 1.09 - max_tongs_height_range
        lift_middle = lift_minimum + max_tongs_height_range / 2.0
    return lift_middle


def get_center_configuration(lift_middle):
    # manipulate lower objects
    center_configuration = {
        "joint_mobile_base_rotation": 0.0,
        "joint_lift": lift_middle,
        "joint_arm_l0": 0.01,
        "joint_wrist_yaw": 0.0,
        "joint_wrist_pitch": 0.0,
        "joint_wrist_roll": 0.0,
    }
    return center_configuration


def get_starting_configuration(lift_middle):

    # The robot will attempt to achieve this configuration before
    # teleoperation begins. Teleoperation commands for the robot's
    # wrist position are made relative to the wrist position
    # associated with this starting configuration.
    starting_configuration = {
        "joint_mobile_base_rotate_by": 0.0,
        "joint_lift": lift_middle,
        "joint_arm_l0": 0.01,
        "joint_wrist_yaw": 0.0,
        "joint_wrist_pitch": 0.0,
        "joint_wrist_roll": 0.0,
        "stretch_gripper": 200.0,
    }
    return starting_configuration


# String used by processes to communicate via shared memory
shared_memory_name = "gripper_goal_20231222"


def goal_dict_to_array(goal_dict):
    if goal_dict is None:
        return None

    goal_array = np.zeros((5, 3), dtype=np.float64)
    grip_width = goal_dict["grip_width"]
    if grip_width is not None:
        goal_array[0, 0] = grip_width
    else:
        goal_array[0, 0] = -10000.0

    goal_array[1, :] = goal_dict["wrist_position"]
    goal_array[2, :] = goal_dict["gripper_x_axis"]
    goal_array[3, :] = goal_dict["gripper_y_axis"]
    goal_array[4, :] = goal_dict["gripper_z_axis"]

    return goal_array


def goal_array_to_dict(goal_array):
    goal_dict = {}
    grip_width = goal_array[0, 0]
    if grip_width < -1000.0:
        grip_width = None
    goal_dict["grip_width"] = grip_width
    goal_dict["wrist_position"] = goal_array[1, :]
    goal_dict["gripper_x_axis"] = goal_array[2, :]
    goal_dict["gripper_y_axis"] = goal_array[3, :]
    goal_dict["gripper_z_axis"] = goal_array[4, :]
    return goal_dict


def get_example_goal_array():

    goal_grip_width = 1.0
    goal_wrist_position = np.array([-0.03, -0.4, 0.9])
    goal_x_axis = np.array([1.0, 0.0, 0.0])
    goal_y_axis = np.array([0.0, -1.0, 0.0])
    goal_z_axis = np.array([0.0, 0.0, -1.0])

    example_goal = {
        "grip_width": goal_grip_width,
        "wrist_position": goal_wrist_position,
        "gripper_x_axis": goal_x_axis,
        "gripper_y_axis": goal_y_axis,
        "gripper_z_axis": goal_z_axis,
    }

    goal_array = goal_dict_to_array(example_goal)
    return goal_array


def get_do_nothing_goal_array():

    goal_grip_width = -10000.0
    goal_wrist_position = np.array([-10000.0, -10000.0, -10000.0])
    goal_x_axis = np.array([1.0, 0.0, 0.0])
    goal_y_axis = np.array([0.0, -1.0, 0.0])
    goal_z_axis = np.array([0.0, 0.0, -1.0])

    example_goal = {
        "grip_width": goal_grip_width,
        "wrist_position": goal_wrist_position,
        "gripper_x_axis": goal_x_axis,
        "gripper_y_axis": goal_y_axis,
        "gripper_z_axis": goal_z_axis,
    }

    goal_array = goal_dict_to_array(example_goal)
    return goal_array


def is_a_do_nothing_goal_array(goal_array):
    test = goal_array[1:, :] < -1000.0
    return np.any(test)
