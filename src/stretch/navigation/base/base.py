# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from abc import ABC
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation


class Pose(ABC):
    """Stores estimated pose from a SLAM backend"""

    def __init__(
        self,
        timestamp: float = 0.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self) -> str:
        return f"timestamp: {self.timestamp}, x: {self.x}, y: {self.y}, z: \
            {self.z}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}"

    def set_timestamp(self, timestamp: float):
        """set pose timestamp (seconds)"""
        self.timestamp = timestamp

    def set_x(self, x: float):
        """set pose x-value (meters)"""
        self.x = x

    def set_y(self, y: float):
        """set pose y-value (meters)"""
        self.y = y

    def set_z(self, z: float):
        """set pose z-value (meters)"""
        self.z = z

    def set_roll(self, roll: float):
        """set pose roll (radians)"""
        self.roll = roll

    def set_pitch(self, pitch: float):
        """set pose pitch (radians)"""
        self.pitch = pitch

    def set_yaw(self, yaw: float):
        """set pose yaw (radians)"""
        self.yaw = yaw

    def get_timestamp(self) -> float:
        """get pose timestamp"""
        return self.timestamp

    def get_x(self) -> float:
        """get pose x-value"""
        return self.x

    def get_y(self) -> float:
        """get pose y-value"""
        return self.y

    def get_z(self) -> float:
        """get pose z-value"""
        return self.z

    def get_roll(self) -> float:
        """get pose roll"""
        return self.roll

    def get_pitch(self) -> float:
        """get pose pitch"""
        return self.pitch

    def get_yaw(self) -> float:
        """get pose yaw"""
        return self.yaw

    def get_rotation_matrix(self) -> np.ndarray:
        """get rotation matrix from euler angles"""
        return Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw]).as_matrix()


class Slam(ABC):
    """slam base class"""

    def __init__(self):
        raise NotImplementedError

    def initialize(self):
        """initialize slam backend"""
        raise NotImplementedError

    def get_pose(self) -> Pose:
        """returns camera pose"""
        raise NotImplementedError

    def get_trajectory_points(self) -> List[Pose]:
        """returns camera trajectory points"""
        raise NotImplementedError

    def start(self):
        """starts slam"""
        raise NotImplementedError
