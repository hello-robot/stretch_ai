# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Tuple

import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation

PI2 = 2 * np.pi


def normalize_ang_error(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % PI2 - np.pi


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles in radians."""
    angle1 = angle1 % PI2
    angle2 = angle2 % PI2
    diff = np.abs(angle1 - angle2)
    return min(diff, PI2 - diff)


def xyt_global_to_base(XYT, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    pose_world2target = xyt2sophus(XYT)
    pose_world2base = xyt2sophus(current_pose)
    pose_base2target = pose_world2base.inverse() * pose_world2target
    return sophus2xyt(pose_base2target)


def xyt_base_to_global(out_XYT, current_pose):
    """
    Transforms the point cloud from base frame into geocentric frame
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    pose_base2target = xyt2sophus(out_XYT)
    pose_world2base = xyt2sophus(current_pose)
    pose_world2target = pose_world2base * pose_base2target
    return sophus2xyt(pose_world2target)


def xyt2sophus(xyt: np.ndarray) -> sp.SE3:
    """
    Converts SE2 coordinates (x, y, rz) to an sophus SE3 pose object.
    """
    x = np.array([xyt[0], xyt[1], 0.0])
    r_mat = sp.SO3.exp([0.0, 0.0, xyt[2]]).matrix()
    return sp.SE3(r_mat, x)


def sophus2xyt(se3: sp.SE3) -> np.ndarray:
    """
    Converts an sophus SE3 pose object to SE2 coordinates (x, y, rz).
    """
    x_vec = se3.translation()
    r_vec = se3.so3().log()
    return np.array([x_vec[0], x_vec[1], r_vec[2]])


def posquat2sophus(pos: Iterable[float], quat: Iterable[float]) -> sp.SE3:
    r_mat = Rotation.from_quat(quat).as_matrix()
    return sp.SE3(r_mat, pos)


def sophus2posquat(se3: sp.SE3) -> Tuple[Iterable[float], Iterable[float]]:
    pos = se3.translation()
    quat = Rotation.from_matrix(se3.so3().matrix()).as_quat()
    return pos, quat


def interpolate_angles(start_angle: float, end_angle: float, step_size: float = 0.1) -> float:
    """Interpolate between two angles in radians with a given step size."""
    start_angle = start_angle % PI2
    end_angle = end_angle % PI2
    diff1 = (end_angle - start_angle) % PI2
    diff2 = (start_angle - end_angle) % PI2
    if diff1 <= diff2:
        direction = 1
        delta = diff1
    else:
        direction = -1
        delta = diff2
    step = min(delta, step_size) * direction
    interpolated_angle = start_angle + step
    return interpolated_angle % PI2


if __name__ == "__main__":
    print(interpolate_angles(4.628, 4.28))
