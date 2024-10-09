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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, List, Optional, Union

import numpy as np

from stretch.core.interfaces import ContinuousNavigationAction
from stretch.motion.robot import RobotModel


class ControlMode(Enum):
    IDLE = 0
    VELOCITY = 1
    NAVIGATION = 2
    MANIPULATION = 3
    BUSY = 4


class AbstractRobotClient(ABC):
    """Connection to a robot."""

    def __init__(self):
        # Init control mode
        self._base_control_mode = ControlMode.IDLE

    @abstractmethod
    def navigate_to(
        self,
        xyt: Union[Iterable[float], ContinuousNavigationAction],
        relative=False,
        blocking=False,
        verbose: bool = False,
        timeout: Optional[float] = None,
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset everything in the robot's internal state"""
        raise NotImplementedError()

    @abstractmethod
    def switch_to_navigation_mode(self):
        raise NotImplementedError()

    @property
    def control_mode(self):
        return self._base_control_mode

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._base_control_mode == ControlMode.MANIPULATION

    def in_navigation_mode(self) -> bool:
        """Is the robot to move around"""
        return self._base_control_mode == ControlMode.NAVIGATION

    def last_motion_failed(self) -> bool:
        """Override this if you want to check to see if a particular motion failed, e.g. it was not reachable and we don't know why."""
        return False

    def start(self) -> bool:
        """Override this if there's custom startup logic that you want to add before anything else.

        Returns True if we actually should do anything (like update) after this."""
        return False

    @abstractmethod
    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        raise NotImplementedError()

    @abstractmethod
    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
        final_timeout: float = 60.0,
    ):
        """Open loop trajectory execution"""
        raise NotImplementedError()

    @abstractmethod
    def move_to_nav_posture(self):
        """Move to a safe posture for navigation"""
        raise NotImplementedError()

    @abstractmethod
    def move_to_manip_posture(self):
        """Move to a safe posture for manipulation"""
        raise NotImplementedError()

    @abstractmethod
    def get_base_pose(self) -> np.ndarray:
        """Get the current pose of the base"""
        raise NotImplementedError()

    @abstractmethod
    def get_pose_graph(self) -> np.ndarray:
        """Get the robot's SLAM pose graph"""
        raise NotImplementedError()

    @abstractmethod
    def at_goal(self) -> bool:
        """Is the robot at a goal?"""
        raise NotImplementedError()

    @abstractmethod
    def save_map(self, filename: str):
        """Save the current map to a file"""
        raise NotImplementedError()

    @abstractmethod
    def load_map(self, filename: str):
        """Load a map from a file"""
        raise NotImplementedError()


class AbstractGraspClient(ABC):
    """Connection to grasping."""

    def set_robot_client(self, robot_client: AbstractRobotClient):
        """Update the robot client this grasping client uses"""
        self.robot_client = robot_client

    def try_grasping(self, object_goal: Optional[str] = None) -> bool:
        raise NotImplementedError
