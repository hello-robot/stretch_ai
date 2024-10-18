# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math

import numpy as np

from stretch.utils.config import get_full_config_path

# Stretch stuff
PLANNER_STRETCH_URDF = get_full_config_path("urdf/planner_calibrated.urdf")
# MANIP_STRETCH_URDF = get_full_config_path("urdf/stretch_manip_mode.urdf")
MANIP_STRETCH_URDF = get_full_config_path("urdf/stretch.urdf")

# This is the gripper, and the distance in the gripper frame to where the fingers will roughly meet
STRETCH_GRASP_FRAME = "link_grasp_center"
STRETCH_CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_BASE_FRAME = "base_link"

# Offsets required for "link_straight_gripper" grasp frame
STRETCH_STANDOFF_DISTANCE = 0.235
STRETCH_STANDOFF_WITH_MARGIN = 0.25
# Offset from a predicted grasp point to STRETCH_GRASP_FRAME
STRETCH_GRASP_OFFSET = np.eye(4)
STRETCH_GRASP_OFFSET[:3, 3] = np.array([0, 0, -1 * STRETCH_STANDOFF_DISTANCE])
# Offset from STRETCH_GRASP_FRAME to predicted grasp point
STRETCH_TO_GRASP = np.eye(4)
STRETCH_TO_GRASP[:3, 3] = np.array([0, 0, STRETCH_STANDOFF_DISTANCE])

# For EXTEND_ARM action
STRETCH_ARM_EXTENSION = 0.8
STRETCH_ARM_LIFT = 0.8

STRETCH_HEAD_CAMERA_ROTATIONS = 3  # number of counterclockwise rotations for the head camera

# For EXTEND_ARM action
STRETCH_ARM_EXTENSION = 0.8
STRETCH_ARM_LIFT = 0.8

look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
look_front = np.array([0.0, -np.pi / 4])
look_ahead = np.array([0.0, 0.0])
look_close = np.array([0.0, math.radians(-45)])
look_down = np.array([0.0, math.radians(-58)])


# Stores joint indices for the Stretch configuration space
class HelloStretchIdx:
    BASE_X = 0
    BASE_Y = 1
    BASE_THETA = 2
    LIFT = 3
    ARM = 4
    GRIPPER = 5
    WRIST_ROLL = 6
    WRIST_PITCH = 7
    WRIST_YAW = 8
    HEAD_PAN = 9
    HEAD_TILT = 10


STRETCH_HOME_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.2,  # lift
        0.057,  # arm
        0.0,  # gripper rpy
        0.0,
        0.0,
        3.0,  # wrist,
        0.0,
        0.0,
    ]
)

# look down in navigation mode for doing manipulation post-navigation
STRETCH_POSTNAV_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        math.radians(-45),
    ]
)

# Gripper pointed down, for a top-down grasp
STRETCH_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)

# Gripper pointed down, for a top-down grasp
STRETCH_DEMO_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.4,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)

# Gripper straight out, lowered arm for clear vision
STRETCH_PREDEMO_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.4,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        0.0,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)
# Navigation should not be fully folded up against the arm - in case its holding something
STRETCH_NAVIGATION_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.6,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        math.radians(-65),
        # look_close[1],
    ]
)


PIN_CONTROLLED_JOINTS = [
    "base_x_joint",
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]

ROS_ARM_JOINTS = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
ROS_LIFT_JOINT = "joint_lift"
ROS_GRIPPER_FINGER = "joint_gripper_finger_left"
# ROS_GRIPPER_FINGER2 = "joint_gripper_finger_right"
ROS_HEAD_PAN = "joint_head_pan"
ROS_HEAD_TILT = "joint_head_tilt"
ROS_WRIST_YAW = "joint_wrist_yaw"
ROS_WRIST_PITCH = "joint_wrist_pitch"
ROS_WRIST_ROLL = "joint_wrist_roll"

stretch_degrees_of_freedom = 3 + 2 + 4 + 2
default_gripper_open_threshold: float = 0.3
