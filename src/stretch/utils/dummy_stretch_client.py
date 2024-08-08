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
from typing import Dict, List, Optional

import numpy as np

from stretch.core import AbstractRobotClient
from stretch.motion import Footprint, RobotModel

# from stretch.motion.kinematics import HelloStretchKinematics


class DummyStretchClient(AbstractRobotClient, RobotModel):
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    def __init__(
        self,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """

        print("warning: dummy client not functional")
        """
        # Robot model
        self._robot_model = HelloStretchKinematics(
            urdf_path=urdf_path,
            ik_type=ik_type,
            visualize=visualize_ik,
            grasp_frame=grasp_frame,
            ee_link_name=ee_link_name,
            manip_mode_controlled_joints=manip_mode_controlled_joints,
        )
        """
        self._robot_model = self

    def navigate_to(self, xyt, relative=False, blocking=False):
        """Move to xyt in global coordinates or relative coordinates."""
        raise NotImplementedError()

    def reset(self):
        """Reset everything in the robot's internal state"""
        self._mode = "navigation"

    def switch_to_navigation_mode(self):
        # return self._base_control_mode == ControlMode.NAVIGATION
        return True

    def get_footprint(self) -> Footprint:
        """Return footprint for the robot. This is expected to be a mask."""
        # Note: close to the actual measurements
        return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        return self._robot_model

    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
    ):
        """Open loop trajectory execution"""
        raise NotImplementedError()
