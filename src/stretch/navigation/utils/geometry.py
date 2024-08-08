# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
from typing import Tuple

import numpy as np

from stretch.navigation.base import Pose


def euler_angles_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> Tuple[float, float, float, float]:
    """
    Convert euler angles into a quaternion.

    Parameters:
    roll (float): Roll angle in radians.
    pitch (float): Pitch angle in radians.
    yaw (float): Yaw angle in radians.

    Returns:
    Tuple[float, float, float, float]: Quaternion (w, x, y, z) components.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return (w, x, y, z)


def quaternion_to_euler_angles(
    w: float, x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """
    Convert a quaternion into euler angles.

    Parameters:
    w (float): Quaternion w component.
    x (float): Quaternion x component.
    y (float): Quaternion y component.
    z (float): Quaternion z component.

    Returns:
    Tuple[float, float, float]: Euler angles (roll, pitch, yaw) in radians.
    """
    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (y-axis rotation)
    sin_pitch = 2 * (w * y - z * x)
    if np.abs(sin_pitch) >= 1:
        pitch = np.sign(sin_pitch) * np.pi / 2
    else:
        pitch = np.arcsin(sin_pitch)

    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return (roll, pitch, yaw)


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a rotation matrix into a quaternion.

    Parameters:
    R (np.ndarray): Rotation matrix.

    Returns:
    Tuple[float, float, float, float]: Quaternion (w, x, y, z) components.
    """
    w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)

    return (w, x, y, z)


def get_pose_in_reference_frame(child_pose: Pose, parent_pose: Pose) -> Pose:
    """
    Compute the pose of a child frame in the reference frame of a parent frame.
    Such that the parent frame is now at the origin (0, 0, 0) and the child
    frame is at a new position.

    Parameters:
    child_pose (Pose): Pose of the child frame.
    parent_pose (Pose): Pose of the parent frame.

    Returns:
    Pose: Pose of the child frame in the reference frame of the parent frame.
    """
    R_parent = parent_pose.get_rotation_matrix()
    t_parent = np.array([parent_pose.x, parent_pose.y, parent_pose.z])
    R_child = child_pose.get_rotation_matrix()
    t_child = np.array([child_pose.x, child_pose.y, child_pose.z])
    t_new = np.dot(R_parent.T, t_child - t_parent)
    R_new = np.dot(R_parent.T, R_child)
    roll, pitch, yaw = quaternion_to_euler_angles(*rotation_matrix_to_quaternion(R_new))

    return Pose(child_pose.timestamp, t_new[0], t_new[1], t_new[2], roll, pitch, yaw)


def transformation_matrix_to_pose(T: np.ndarray) -> Pose:
    """
    Convert a transformation matrix into a Pose object.

    Parameters:
    T (np.ndarray): Transformation matrix.

    Returns:
    Pose: Pose object.
    """
    roll = np.arctan2(T[2, 1], T[2, 2])
    pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
    yaw = np.arctan2(T[1, 0], T[0, 0])

    return Pose(0, T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw)
