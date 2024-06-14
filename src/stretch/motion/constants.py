import math

import numpy as np

# Stretch stuff
PLANNER_STRETCH_URDF = "config/urdf/planner_calibrated.urdf"
MANIP_STRETCH_URDF = "config/urdf/stretch_manip_mode.urdf"

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
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        math.radians(-30),
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
