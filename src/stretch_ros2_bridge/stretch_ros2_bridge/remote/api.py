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
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import trimesh.transformations as tra

from stretch.core.interfaces import Observations
from stretch.core.robot import AbstractRobotClient, ControlMode
from stretch.motion import RobotModel
from stretch.motion.constants import STRETCH_NAVIGATION_Q, STRETCH_PREGRASP_Q
from stretch.motion.kinematics import HelloStretchIdx, HelloStretchKinematics
from stretch.utils.geometry import xyt2sophus

from .modules.head import StretchHeadClient
from .modules.manip import StretchManipulationClient
from .modules.mapping import StretchMappingClient
from .modules.nav import StretchNavigationClient
from .ros import StretchRosInterface


class StretchClient(AbstractRobotClient):
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    head_camera_frame = "camera_color_optical_frame"
    ee_camera_frame = "gripper_camera_color_optical_frame"
    ee_frame = "link_grasp_center"
    world_frame = "map"

    def __init__(
        self,
        init_node: bool = True,
        camera_overrides: Optional[Dict] = None,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
        d405: bool = True,
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """

        if camera_overrides is None:
            camera_overrides = {}
        self._ros_client = StretchRosInterface(init_lidar=True, d405=d405, **camera_overrides)

        # Robot model
        self._robot_model = HelloStretchKinematics(
            urdf_path=urdf_path,
            ik_type=ik_type,
            visualize=visualize_ik,
            grasp_frame=grasp_frame,
            ee_link_name=ee_link_name,
            manip_mode_controlled_joints=manip_mode_controlled_joints,
        )

        # Interface modules
        self.nav = StretchNavigationClient(self._ros_client, self._robot_model)
        self.manip = StretchManipulationClient(self._ros_client, self._robot_model)
        self.head = StretchHeadClient(self._ros_client, self._robot_model)
        self.mapping = StretchMappingClient(self._ros_client)

        # Init control mode
        self._base_control_mode = ControlMode.IDLE

        # Initially start in navigation mode all the time - in order to make sure we are initialized into a decent state. Otherwise we need to check the different components and safely figure out control mode, which can be inaccurate.
        self.switch_to_navigation_mode()

    @property
    def model(self):
        return self._robot_model

    @property
    def is_homed(self) -> bool:
        return self._ros_client.is_homed

    @property
    def is_runstopped(self) -> bool:
        return self._ros_client.is_runstopped

    def at_goal(self) -> bool:
        """Returns true if we have up to date head info and are at goal position"""
        return self.nav.at_goal()

    # Mode interfaces

    def switch_to_navigation_mode(self):
        """Switch stretch to navigation control
        Robot base is now controlled via continuous velocity feedback.
        """
        result_pre = True
        if self.manip.is_enabled:
            result_pre = self.manip.disable()

        result_post = self.nav.enable()

        self._base_control_mode = ControlMode.NAVIGATION

        return result_pre and result_post

    @property
    def base_control_mode(self) -> ControlMode:
        return self._base_control_mode

    def switch_to_busy_mode(self) -> bool:
        """Switch to a mode that says we are occupied doing something blocking"""
        self._base_control_mode = ControlMode.BUSY
        return True

    def switch_to_manipulation_mode(self):
        """Switch stretch to manipulation control
        Robot base is now controlled via position control.
        Base rotation is locked.
        """
        result_pre = True
        if self.nav.is_enabled:
            result_pre = self.nav.disable()

        result_post = self.manip.enable()

        self._base_control_mode = ControlMode.MANIPULATION

        return result_pre and result_post

    # General control methods

    def wait(self):
        self.nav.wait()
        self.manip.wait()
        self.head.wait()

    def reset(self):
        self.stop()
        self.switch_to_manipulation_mode()
        self.manip.home()
        self.switch_to_navigation_mode()
        self.nav.home()
        self.stop()

    def stop(self):
        """Stop the robot"""
        self.nav.disable()
        self.manip.disable()
        self._base_control_mode = ControlMode.IDLE

    # Other interfaces

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning. Overrides base class method"""
        return self._robot_model

    def get_ros_client(self) -> StretchRosInterface:
        """return the internal ROS client"""
        return self._ros_client

    @property
    def robot_joint_pos(self):
        return self._ros_client.pos

    @property
    def camera_pose(self):
        return self.head_camera_pose

    @property
    def head_camera_pose(self):
        p0 = self._ros_client.get_frame_pose(
            self.head_camera_frame, base_frame=self.world_frame, timeout_s=5.0
        )
        if p0 is not None:
            p0 = p0 @ tra.euler_matrix(0, 0, -np.pi / 2)
        return p0

    @property
    def ee_camera_pose(self):
        p0 = self._ros_client.get_frame_pose(
            self.ee_camera_frame, base_frame=self.world_frame, timeout_s=5.0
        )
        return p0

    @property
    def ee_pose(self):
        p0 = self._ros_client.get_frame_pose(self.ee_frame, base_frame=self.world_frame)
        if p0 is not None:
            p0 = p0 @ tra.euler_matrix(0, 0, 0)
        return p0

    @property
    def rgb_cam(self):
        return self._ros_client.rgb_cam

    @property
    def dpt_cam(self):
        return self._ros_client.dpt_cam

    @property
    def ee_dpt_cam(self):
        return self._ros_client.ee_dpt_cam

    @property
    def ee_rgb_cam(self):
        return self._ros_client.ee_rgb_cam

    @property
    def lidar(self):
        return self._ros_client._lidar

    def get_joint_state(self):
        """Get joint states from the robot. If in manipulation mode, use the base_x position from start of manipulation mode as the joint state for base_x."""
        q, dq, eff = self._ros_client.get_joint_state()
        # If we are in manipulation mode...
        if self._base_control_mode == ControlMode.MANIPULATION:
            # ...we need to get the joint positions from the manipulator
            q[HelloStretchIdx.BASE_X] = self.manip.get_base_x()
        return q, dq, eff

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None):
        """look up a particular frame in base coords"""
        return self._ros_client.get_frame_pose(frame, base_frame, lookup_time)

    def move_to_manip_posture(self):
        """Move the arm and head into manip mode posture: gripper down, head facing the gripper."""
        self.switch_to_manipulation_mode()
        pos = self.manip._extract_joint_pos(STRETCH_PREGRASP_Q)
        pan, tilt = self._robot_model.look_at_ee
        print("- go to configuration:", pos, "pan =", pan, "tilt =", tilt)
        self.manip.goto_joint_positions(pos, head_pan=pan, head_tilt=tilt, blocking=True)
        print("- Robot switched to manipulation mode.")

    def move_to_nav_posture(self):
        """Move the arm and head into nav mode. The head will be looking front."""

        # First retract the robot's joints
        self.switch_to_manipulation_mode()
        pan, tilt = self._robot_model.look_close
        pos = self.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
        print("- go to configuration:", pos, "pan =", pan, "tilt =", tilt)
        self.manip.goto_joint_positions(pos, head_pan=pan, head_tilt=tilt, blocking=True)
        self.switch_to_navigation_mode()
        print("- Robot switched to navigation mode.")

    def get_base_pose(self) -> np.ndarray:
        """Get the robot's base pose as XYT."""
        return self.nav.get_base_pose()

    def get_pose_graph(self) -> np.ndarray:
        """Get SLAM pose graph as a numpy array"""
        graph = self._ros_client.get_pose_graph()
        for i in range(len(graph)):
            relative_pose = xyt2sophus(np.array(graph[i][1:]))
            euler_angles = relative_pose.so3().log()
            theta = euler_angles[-1]

            # GPS in robot coordinates
            gps = relative_pose.translation()[:2]

            graph[i] = np.array([graph[i][0], gps[0], gps[1], theta])

        return graph

    def load_map(self, filename: str):
        self.mapping.load_map(filename)

    def save_map(self, filename: str):
        self.mapping.save_map(filename)

    def execute_trajectory(self, *args, **kwargs):
        """Open-loop trajectory execution wrapper. Executes a multi-step trajectory; this is always blocking since it waits to reach each one in turn."""
        return self.nav.execute_trajectory(*args, **kwargs)

    def navigate_to(
        self,
        xyt: Iterable[float],
        relative: bool = False,
        blocking: bool = True,
    ):
        """
        Move to xyt in global coordinates or relative coordinates. Cannot be used in manipulation mode.
        """
        return self.nav.navigate_to(xyt, relative=relative, blocking=blocking)

    def get_observation(
        self,
        rotate_head_pts=False,
        start_pose: Optional[np.ndarray] = None,
        compute_xyz: bool = True,
    ) -> Observations:
        """Get an observation from the current robot.

        Parameters:
            rotate_head_pts: this is true to put things into the same format as Habitat; generally we do not want to do this
        """

        # Computing XYZ is expensive, we do not always needd to do it
        if compute_xyz:
            rgb, depth, xyz = self.head.get_images(compute_xyz=True)
        else:
            rgb, depth = self.head.get_images(compute_xyz=False)
            xyz = None

        current_pose = xyt2sophus(self.nav.get_base_pose())

        if start_pose is not None:
            # use sophus to get the relative translation
            relative_pose = start_pose.inverse() * current_pose
        else:
            relative_pose = current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Get joint state information
        joint_positions, _, _ = self.get_joint_state()

        # Get lidar points and timestamp
        lidar_points = self.lidar.get()
        lidar_timestamp = self.lidar.get_time().nanoseconds / 1e9

        # Create the observation
        obs = Observations(
            rgb=rgb,
            depth=depth,
            xyz=xyz,
            gps=gps,
            compass=np.array([theta]),
            camera_pose=self.head_camera_pose,
            joint=joint_positions,
            camera_K=self.get_camera_intrinsics(),
            lidar_points=lidar_points,
            lidar_timestamp=lidar_timestamp,
        )
        return obs

    def get_camera_intrinsics(self) -> torch.Tensor:
        """Get 3x3 matrix of camera intrisics K"""
        return torch.from_numpy(self.head._ros_client.rgb_cam.K).float()

    def head_to(self, pan: float, tilt: float, blocking: bool = False):
        """Send head commands"""
        self.head.goto_joint_positions(pan=float(pan), tilt=float(tilt), blocking=blocking)

    def arm_to(
        self,
        q: np.ndarray,
        gripper: float = None,
        head_pan: float = None,
        head_tilt: float = None,
        blocking: bool = False,
    ):
        """Send arm commands"""
        assert len(q) == 6

        print(f"-> Sending arm and gripper to {q=} {gripper=} {head_pan=} {head_tilt=}")

        self.manip.goto_joint_positions(
            joint_positions=q,
            gripper=gripper,
            head_pan=head_pan,
            head_tilt=head_tilt,
            blocking=blocking,
        )


if __name__ == "__main__":
    import rclpy

    rclpy.init()
    client = StretchClient()
    breakpoint()
