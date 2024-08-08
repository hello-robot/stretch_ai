# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh


class GeneralTaskState(Enum):
    NOT_STARTED = 0
    PREPPING = 1
    DOING_TASK = 2
    IDLE = 3
    STOP = 4


class Action:
    """Controls."""

    pass


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_OBJECT = 4
    PLACE_OBJECT = 5
    NAVIGATION_MODE = 6
    MANIPULATION_MODE = 7
    POST_NAV_MODE = 8
    # Arm extension to a fixed position and height
    EXTEND_ARM = 9
    EMPTY_ACTION = 10
    # Simulation only actions
    SNAP_OBJECT = 11
    DESNAP_OBJECT = 12
    # Discrete gripper commands
    OPEN_GRIPPER = 13
    CLOSE_GRIPPER = 14


class ContinuousNavigationAction(Action):
    xyt: np.ndarray

    def __init__(self, xyt: np.ndarray):
        if not len(xyt) == 3:
            raise RuntimeError("continuous navigation action space has 3 dimensions, x y and theta")
        self.xyt = xyt

    def __str__(self):
        return f"xyt={self.xyt}"


class ContinuousFullBodyAction:
    xyt: np.ndarray
    joints: np.ndarray

    def __init__(self, joints: np.ndarray, xyt: np.ndarray = None):
        """Create full-body continuous action"""
        if xyt is not None and not len(xyt) == 3:
            raise RuntimeError("continuous navigation action space has 3 dimensions, x y and theta")
        self.xyt = xyt
        # Joint states in robot action format
        self.joints = joints


class ContinuousEndEffectorAction:
    pos: np.ndarray
    ori: np.ndarray
    g: np.ndarray
    num_actions: int

    def __init__(
        self,
        pos: np.ndarray = None,
        ori: np.ndarray = None,
        g: np.ndarray = None,
    ):
        """Create end-effector continuous action; moves to 6D pose and activates gripper"""
        if (
            pos is not None
            and ori is not None
            and g is not None
            and not (pos.shape[1] + ori.shape[1] + g.shape[1]) == 8
        ):
            raise RuntimeError(
                "continuous end-effector action space has 8 dimensions: pos=3, ori=4, gripper=1"
            )
        self.pos = pos
        self.ori = ori
        self.g = g
        self.num_actions = pos.shape[0]


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS_NAVIGATION = 1
    CONTINUOUS_MANIPULATION = 2
    CONTINUOUS_EE_MANIPULATION = 3


class HybridAction(Action):
    """Convenience for supporting multiple action types - provides handling to make sure we have the right class at any particular time"""

    action_type: ActionType
    action: Action

    def __init__(
        self,
        action=None,
        xyt: np.ndarray = None,
        joints: np.ndarray = None,
        pos: np.ndarray = None,
        ori: np.ndarray = None,
        gripper: np.ndarray = None,
    ):
        """Make sure that we were passed a useful generic action here. Process it into something useful."""
        if action is not None:
            if isinstance(action, HybridAction):
                self.action_type = action.action_type
            if isinstance(action, DiscreteNavigationAction):
                self.action_type = ActionType.DISCRETE
            elif isinstance(action, ContinuousNavigationAction):
                self.action_type = ActionType.CONTINUOUS_NAVIGATION
            elif isinstance(action, ContinuousEndEffectorAction):
                self.action_type = ActionType.CONTINUOUS_EE_MANIPULATION
            else:
                self.action_type = ActionType.CONTINUOUS_MANIPULATION
        elif joints is not None:
            self.action_type = ActionType.CONTINUOUS_MANIPULATION
            action = ContinuousFullBodyAction(joints, xyt)
        elif xyt is not None:
            self.action_type = ActionType.CONTINUOUS_NAVIGATION
            action = ContinuousNavigationAction(xyt)
        elif pos is not None:
            self.action_type = ActionType.CONTINUOUS_EE_MANIPULATION
            action = ContinuousEndEffectorAction(pos, ori, gripper)
        else:
            raise RuntimeError("Cannot create HybridAction without any action!")
        if isinstance(action, HybridAction):
            # TODO: should we copy like this?
            self.action_type = action.action_type
            action = action.action
            # But more likely this was a mistake so let's actually throw an error
            raise RuntimeError("Do not pass a HybridAction when creating another HybridAction!")
        self.action = action

    def is_discrete(self):
        """Let environment know if we need to handle a discrete action"""
        return self.action_type == ActionType.DISCRETE

    def is_navigation(self):
        return self.action_type == ActionType.CONTINUOUS_NAVIGATION

    def is_manipulation(self):
        return self.action_type in [
            ActionType.CONTINUOUS_MANIPULATION,
            ActionType.CONTINUOUS_EE_MANIPULATION,
        ]

    def get(self):
        """Extract continuous component of the command and return it."""
        if self.action_type == ActionType.DISCRETE:
            return self.action
        elif self.action_type == ActionType.CONTINUOUS_NAVIGATION:
            return self.action.xyt
        elif self.action_type == ActionType.CONTINUOUS_EE_MANIPULATION:
            return self.action.pos, self.action.ori, self.action.g
        else:
            # Extract both the joints and the waypoint target
            return self.action.joints, self.action.xyt


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class Observations:
    """Sensor observations."""

    # --------------------------------------------------------
    # Typed observations
    # --------------------------------------------------------

    # Joint states
    # joint_positions: np.ndarray

    # Pose
    # TODO: add these instead of gps + compass
    # base_pose: Pose
    # ee_pose: Pose

    # Pose
    gps: np.ndarray  # (x, y) where positive x is forward, positive y is translation to left in meters
    compass: np.ndarray  # positive theta is rotation to left in radians - consistent with robot

    # Camera
    rgb: np.ndarray  # (camera_height, camera_width, 3) in [0, 255]
    depth: np.ndarray  # (camera_height, camera_width) in meters
    xyz: Optional[np.ndarray] = None  # (camera_height, camera_width, 3) in camera coordinates
    semantic: Optional[
        np.ndarray
    ] = None  # (camera_height, camera_width) in [0, num_sem_categories - 1]
    camera_K: Optional[np.ndarray] = None  # (3, 3) camera intrinsics matrix

    # Pose of the camera in world coordinates
    camera_pose: Optional[np.ndarray] = None

    # End effector camera
    ee_rgb: Optional[np.ndarray] = None  # (camera_height, camera_width, 3) in [0, 255]
    ee_depth: Optional[np.ndarray] = None  # (camera_height, camera_width) in meters
    ee_xyz: Optional[np.ndarray] = None  # (camera_height, camera_width, 3) in camera coordinates
    ee_semantic: Optional[
        np.ndarray
    ] = None  # (camera_height, camera_width) in [0, num_sem_categories - 1]
    ee_camera_K: Optional[np.ndarray] = None  # (3, 3) camera intrinsics matrix

    # Pose of the end effector camera in world coordinates
    ee_camera_pose: Optional[np.ndarray] = None

    # Pose of the end effector grasp center in world coordinates
    ee_pose: Optional[np.ndarray] = None

    # Instance IDs per observation frame
    # Size: (camera_height, camera_width)
    # Range: 0 to max int
    instance: Optional[np.ndarray] = None

    # Optional third-person view from simulation
    third_person_image: Optional[np.ndarray] = None

    # Proprioreception
    joint: Optional[np.ndarray] = None  # joint positions of the robot
    relative_resting_position: Optional[
        np.ndarray
    ] = None  # end-effector position relative to the desired resting position
    is_holding: Optional[np.ndarray] = None  # whether the agent is holding the object
    # --------------------------------------------------------
    # Untyped task-specific observations
    # --------------------------------------------------------

    task_observations: Optional[Dict[str, Any]] = None

    # Sequence number - which message was this?
    seq_id: int = -1

    def compute_xyz(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Compute xyz from depth and camera intrinsics."""
        if self.depth is not None and self.camera_K is not None:
            self.xyz = self.depth_to_xyz(self.depth * scaling, self.camera_K)
        return self.xyz

    def compute_ee_xyz(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Compute xyz from depth and camera intrinsics."""
        if self.ee_depth is not None and self.ee_camera_K is not None:
            self.ee_xyz = self.depth_to_xyz(self.ee_depth * scaling, self.ee_camera_K)
        return self.ee_xyz

    def depth_to_xyz(self, depth, camera_K) -> np.ndarray:
        """Convert depth image to xyz point cloud."""
        # Get the camera intrinsics
        fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
        # Get the image size
        h, w = depth.shape
        # Create the grid
        x = np.tile(np.arange(w), (h, 1))
        y = np.tile(np.arange(h).reshape(-1, 1), (1, w))
        # Compute the xyz
        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        return np.stack([x, y, depth], axis=-1)

    def get_ee_xyz_in_world_frame(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Get the end effector xyz in world frame."""
        if self.ee_xyz is None:
            self.compute_ee_xyz(scaling=scaling)
        if self.ee_xyz is not None and self.ee_camera_pose is not None:
            return self.transform_points(self.ee_xyz, self.ee_camera_pose)
        return None

    def get_xyz_in_world_frame(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Get the xyz in world frame."""
        if self.xyz is None:
            self.compute_xyz(scaling=scaling)
        if self.xyz is not None and self.camera_pose is not None:
            return self.transform_points(self.xyz, self.camera_pose)
        return None

    def transform_points(self, points: np.ndarray, pose: np.ndarray):
        """Transform points to world frame."""
        return np.dot(points, pose[:3, :3].T) + pose[:3, 3]
