# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import errno
import os

import numpy as np
import urchin as urdf_loader
from scipy.spatial.transform import Rotation

from stretch.app.dex_teleop.dex_teleop_parameters import (
    DEX_TELEOP_CONTROLLED_JOINTS,
    SUPPORTED_MODES,
)
from stretch.utils.geometry import get_rotation_from_xyz


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


def format_state(raw_state: dict | None = None, teleop_mode: str | None = "base_x"):
    """
    This function is used with the old leader-follower with stretch_body move_to backend.
    State format: [x, x_vel, y, y_vel, theta, theta_vel, lift, arm, wrist_r, wrist_p, wrist_y, gripper]
    """
    # Zero out unused features for specific teleop modes
    if teleop_mode == "standard":
        pass
    elif teleop_mode == "rotary_base":
        raw_state["base_x"] = 0.0
        raw_state["base_x_vel"] = 0.0
        raw_state["base_y"] = 0.0
        raw_state["base_y_vel"] = 0.0

    elif teleop_mode == "stationary_base":
        raw_state["base_x"] = 0.0
        raw_state["base_x_vel"] = 0.0
        raw_state["base_y"] = 0.0
        raw_state["base_y_vel"] = 0.0
        raw_state["base_theta"] = 0.0
        raw_state["base_theta_vel"] = 0.0

    elif teleop_mode == "base_x":
        raw_state["base_y"] = 0.0
        raw_state["base_y_vel"] = 0.0
        raw_state["base_theta"] = 0.0
        raw_state["base_theta_vel"] = 0.0

    # TODO Remove once support for older models with feature dim of 7 isn't needed
    elif teleop_mode == "old_stationary_base":
        pass
    else:
        raise NotImplementedError(
            f"{teleop_mode} is not a supported teleop mode. Supported modes: {SUPPORTED_MODES}"
        )

    return raw_state


def format_actions(raw_actions: dict):
    """Format actions: remove unused actions depending on controlled joints and set them to zero."""

    if raw_actions is None:
        return None

    for x in DEX_TELEOP_CONTROLLED_JOINTS:
        if x not in raw_actions:
            raw_actions[x] = 0.0

    # Remove individual arm joints
    del raw_actions["joint_arm_l1"]
    del raw_actions["joint_arm_l2"]
    del raw_actions["joint_arm_l3"]

    return raw_actions


def get_teleop_controlled_joints(teleop_mode: str):
    arm = [
        "joint_arm_l0",
        "joint_lift",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]
    if teleop_mode == "base_x":
        return arm.append("base_x_joint")
    elif teleop_mode == "rotary_base":
        return arm.append("base_theta_joint")
    elif teleop_mode == "stationary_base":
        return arm


def process_goal_dict(
    goal_dict: dict, prev_goal_dict: dict = None, use_gripper_center: bool = True
) -> dict:
    """Process goal dict:
    - fix orientation if necessary
    - calculate relative gripper position and orientation
    - compute quaternion
    - fix offsets if necessary
    """

    if "gripper_x_axis" not in goal_dict:
        # If we don't have the necessary information, return the goal_dict as is
        # This means tool was not detected
        goal_dict["valid"] = False
        return goal_dict

    # Convert goal dict into a quaternion
    # Start by getting the rotation as a usable object
    r = get_rotation_from_xyz(
        goal_dict["gripper_x_axis"],
        goal_dict["gripper_y_axis"],
        goal_dict["gripper_z_axis"],
    )
    if use_gripper_center:
        # Apply conversion
        # This is a simple frame transformation which should rotate into gripper grasp frame
        delta = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        r_matrix = r.as_matrix() @ delta
        r = r.from_matrix(r_matrix)
    else:
        goal_dict["gripper_orientation"] = r.as_quat()
        r_matrix = r.as_matrix()

    # Get pose matrix for current frame
    T1 = np.eye(4)
    T1[:3, :3] = r_matrix
    T1[:3, 3] = goal_dict["wrist_position"]

    if use_gripper_center:
        T_wrist_to_grasp = np.eye(4)
        T_wrist_to_grasp[2, 3] = 0
        T_wrist_to_grasp[0, 3] = 0.3
        # T_wrist_to_grasp[1, 3] = 0.3
        T1 = T1 @ T_wrist_to_grasp
        goal_dict["gripper_orientation"] = Rotation.from_matrix(T1[:3, :3]).as_quat()
        goal_dict["gripper_x_axis"] = T1[:3, 0]
        goal_dict["gripper_y_axis"] = T1[:3, 1]
        goal_dict["gripper_z_axis"] = T1[:3, 2]
        goal_dict["wrist_position"] = T1[:3, 3]
        # Note: print debug information; TODO: remove
        # If we need to, we can tune this to center the gripper
        # Charlie had some code which did this in a slightly nicer way, I think
        # print(T1[:3, 3])

    goal_dict["use_gripper_center"] = use_gripper_center
    if prev_goal_dict is not None and "gripper_orientation" in prev_goal_dict:

        T0 = np.eye(4)
        r0 = Rotation.from_quat(prev_goal_dict["gripper_orientation"])
        T0[:3, :3] = r0.as_matrix()
        T0[:3, 3] = prev_goal_dict["wrist_position"]

        T = np.linalg.inv(T0) @ T1
        goal_dict["relative_gripper_position"] = T[:3, 3]
        goal_dict["relative_gripper_orientation"] = Rotation.from_matrix(T[:3, :3]).as_quat()

        goal_dict["valid"] = True
    else:
        goal_dict["valid"] = False

    return goal_dict
