# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Dict, List, Tuple

import numpy as np

from stretch.motion import HelloStretchIdx

HelloStretchManipIdx: Dict[str, int] = {
    "BASE_X": 0,
    "LIFT": 1,
    "ARM": 2,
    "WRIST_YAW": 3,
    "WRIST_PITCH": 4,
    "WRIST_ROLL": 5,
}


def delta_hab_to_position_command(cmd, pan, tilt, deltas) -> Tuple[List[float], float, float]:
    """Compute deltas"""
    assert len(deltas) == 10
    arm = deltas[0] + deltas[1] + deltas[2] + deltas[3]
    lift = deltas[4]
    roll = deltas[5]
    pitch = deltas[6]
    yaw = deltas[7]
    positions = [
        0,  # This is the robot's base x axis - not currently used
        cmd[1] + lift,
        cmd[2] + arm,
        cmd[3] + yaw,
        cmd[4] + pitch,
        cmd[5] + roll,
    ]
    pan = pan + deltas[8]
    tilt = tilt + deltas[9]
    return positions, pan, tilt


def config_to_manip_command(q):
    """convert from general representation into arm manip command. This extracts just the information used for end-effector control: base x motion, arm lift, and wrist variables."""
    return [
        q[HelloStretchIdx.BASE_X],
        q[HelloStretchIdx.LIFT],
        q[HelloStretchIdx.ARM],
        q[HelloStretchIdx.WRIST_YAW],
        q[HelloStretchIdx.WRIST_PITCH],
        q[HelloStretchIdx.WRIST_ROLL],
    ]


def config_to_hab(q: np.ndarray) -> np.ndarray:
    """Convert default configuration into habitat commands. This is a slightly different format that strips out x, y, and theta."""
    hab = np.zeros(10)
    hab[0] = q[HelloStretchIdx.ARM]
    hab[4] = q[HelloStretchIdx.LIFT]
    hab[5] = q[HelloStretchIdx.WRIST_ROLL]
    hab[6] = q[HelloStretchIdx.WRIST_PITCH]
    hab[7] = q[HelloStretchIdx.WRIST_YAW]
    hab[8] = q[HelloStretchIdx.HEAD_PAN]
    hab[9] = q[HelloStretchIdx.HEAD_TILT]
    return hab


def hab_to_position_command(hab_positions) -> Tuple[List[float], float, float]:
    """Compute hab_positions"""
    assert len(hab_positions) == 10
    arm = hab_positions[0] + hab_positions[1] + hab_positions[2] + hab_positions[3]
    lift = hab_positions[4]
    roll = hab_positions[5]
    pitch = hab_positions[6]
    yaw = hab_positions[7]
    positions = [
        0,  # This is the robot's base x axis - not currently used
        lift,
        arm,
        yaw,
        pitch,
        roll,
    ]
    pan = hab_positions[8]
    tilt = hab_positions[9]
    return positions, pan, tilt


def get_manip_joint_idx(joint: str) -> int:
    """Get manip joint index"""
    return HelloStretchManipIdx[joint.upper()]
