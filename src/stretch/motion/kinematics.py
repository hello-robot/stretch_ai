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
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from stretch.core.interfaces import ContinuousFullBodyAction
from stretch.motion.base import IKSolverBase
from stretch.motion.constants import (
    MANIP_STRETCH_URDF,
    PLANNER_STRETCH_URDF,
    STRETCH_GRASP_FRAME,
    STRETCH_HOME_Q,
)
from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver, PositionIKOptimizer
from stretch.motion.robot import Footprint


# used for mapping joint states in STRETCH_*_Q to match the sim/real joint action space
def map_joint_q_state_to_action_space(q):
    return np.array(
        [
            q[4],  # arm_0
            q[3],  # lift
            q[8],  # yaw
            q[7],  # pitch
            q[6],  # roll
            q[9],  # head pan
            q[10],  # head tilt
        ]
    )


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

    name_to_idx = {
        "base_x": BASE_X,
        "base_y": BASE_Y,
        "base_theta": BASE_THETA,
        "lift": LIFT,
        "arm": ARM,
        "gripper_finger_right": GRIPPER,
        "gripper": GRIPPER,
        "wrist_roll": WRIST_ROLL,
        "wrist_pitch": WRIST_PITCH,
        "wrist_yaw": WRIST_YAW,
        "head_pan": HEAD_PAN,
        "head_tilt": HEAD_TILT,
    }

    @classmethod
    def get_idx(cls, name: str) -> int:
        if name in cls.name_to_idx:
            return cls.name_to_idx[name]
        else:
            raise ValueError(f"Unknown joint name: {name}")


class HelloStretchKinematics:
    """Define motion planning structure for the robot. Exposes kinematics."""

    # DEFAULT_BASE_HEIGHT = 0.09
    DEFAULT_BASE_HEIGHT = 0
    GRIPPER_OPEN = 0.6
    GRIPPER_CLOSED = -0.3
    GRIPPER_CLOSED_LOOSE = 0.0

    default_step = np.array(
        [
            0.1,
            0.1,
            0.2,  # x y theta
            0.025,
            0.025,  # lift and arm
            0.3,  # gripper
            0.1,
            0.1,
            0.1,  # wrist rpy
            0.2,
            0.2,  # head
        ]
    )
    default_tols = np.array(
        [
            0.1,
            0.1,
            0.01,  # x y theta
            0.001,
            0.0025,  # lift and arm
            0.01,  # gripper
            0.01,
            0.01,
            0.01,  # wrist rpy
            10.0,
            10.0,  # head - TODO handle this better
        ]
    )
    # look_at_ee = np.array([-np.pi/2, -np.pi/8])
    look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
    look_front = np.array([0.0, math.radians(-30)])
    look_ahead = np.array([0.0, 0.0])
    look_close = np.array([0.0, math.radians(-45)])
    look_down = np.array([0.0, math.radians(-58)])

    max_arm_height = 1.2

    # For inverse kinematics mode
    default_ee_link_name = "link_grasp_center"

    default_manip_mode_controlled_joints = [
        # "base_x_joint",
        "joint_fake",
        "joint_lift",
        "joint_arm_l3",
        "joint_arm_l2",
        "joint_arm_l1",
        "joint_arm_l0",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]
    manip_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10]
    full_body_controlled_joints = [
        "base_x_joint",
        "base_y_joint",
        "base_theta_joint",
        "joint_lift",
        "joint_arm_l3",
        "joint_arm_l2",
        "joint_arm_l1",
        "joint_arm_l0",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]

    def get_footprint(self) -> Footprint:
        """Return footprint for the robot. This is expected to be a mask."""
        # Note: close to the actual measurements
        return Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)
        # return Footprint(width=0.4, length=0.5, width_offset=0.0, length_offset=0.1)
        # return Footprint(width=0.2, length=0.2, width_offset=0.0, length_offset=0.1)

    def _create_ik_solvers(self, ik_type: str = "pinocchio", visualize: bool = False):
        """Create ik solvers using physics backends such as pinocchio."""
        # You can set one of the visualize flags to true to debug IK issues
        # This is not exposed manually - only one though or it will fail
        assert ik_type in [
            "pinocchio",
            "pinocchio_optimize",
        ], f"Unknown ik type: {ik_type}"

        # You can set one of the visualize flags to true to debug IK issues
        self._manip_dof = len(self._manip_mode_controlled_joints)
        _manip_ik_solver = PinocchioIKSolver(
            self.manip_mode_urdf_path,
            self._ee_link_name,
            self._manip_mode_controlled_joints,
        )
        self.manip_ik_solver: Optional[IKSolverBase] = None
        if "optimize" in ik_type:
            self.manip_ik_solver = PositionIKOptimizer(
                ik_solver=_manip_ik_solver,
                pos_error_tol=0.005,
                ori_error_range=np.array([0.0, 0.0, 0.2]),
            )
        else:
            self.manip_ik_solver = _manip_ik_solver

    def __init__(
        self,
        name: str = "hello_robot_stretch",
        urdf_path: str = "",
        visualize: bool = False,
        root: str = ".",
        ik_type: str = "pinocchio",
        ee_link_name: Optional[str] = None,
        grasp_frame: Optional[str] = None,
        joint_tolerance: float = 0.01,
        manip_mode_controlled_joints: Optional[List[str]] = None,
    ):
        """Create the robot in bullet for things like kinematics; extract information"""

        self.joint_tol = joint_tolerance

        # urdf
        if not urdf_path:
            full_body_urdf = PLANNER_STRETCH_URDF
            manip_urdf = MANIP_STRETCH_URDF
        else:
            full_body_urdf = os.path.join(urdf_path, "planner_calibrated.urdf")
            manip_urdf = os.path.join(urdf_path, "planner_calibrated_manipulation_mode.urdf")
        self.full_body_urdf_path = os.path.join(root, full_body_urdf)
        self.manip_mode_urdf_path = os.path.join(root, manip_urdf)
        self.name = name
        self.visualize = visualize

        # DOF: 3 for ee roll/pitch/yaw
        #      1 for gripper
        #      1 for ee extension
        #      3 for base x/y/theta
        #      2 for head
        self.dof = 3 + 2 + 4 + 2
        self.joints_dof = 10  # from habitat spec
        self.base_height = self.DEFAULT_BASE_HEIGHT

        # ranges for joints
        self.range = np.zeros((self.dof, 2))

        # Create object reference
        # self.set_pose = self.ref.set_pose
        # self.set_joint_position = self.ref.set_joint_position

        self._ik_type = ik_type
        self._ee_link_name = ee_link_name if ee_link_name is not None else self.default_ee_link_name
        self._grasp_frame = grasp_frame if grasp_frame is not None else STRETCH_GRASP_FRAME
        self._manip_mode_controlled_joints = (
            manip_mode_controlled_joints
            if manip_mode_controlled_joints is not None
            else self.default_manip_mode_controlled_joints
        )

        self._create_ik_solvers(ik_type=ik_type, visualize=visualize)

    def get_dof(self) -> int:
        """return degrees of freedom of the robot"""
        return self.dof

    def sample_uniform(self, q0=None, pos=None, radius=2.0):
        """Sample random configurations to seed the ik planner"""
        q = (np.random.random(self.dof) * self._rngs) + self._mins
        q[HelloStretchIdx.BASE_THETA] = np.random.random() * np.pi * 2
        # Set the gripper state
        if q0 is not None:
            q[HelloStretchIdx.GRIPPER] = q0[HelloStretchIdx.GRIPPER]
        # Set the position to sample poses
        if pos is not None:
            x, y = pos[0], pos[1]
        elif q0 is not None:
            x = q0[HelloStretchIdx.BASE_X]
            y = q0[HelloStretchIdx.BASE_Y]
        else:
            x, y = None, None
        # Randomly sample
        if x is not None:
            theta = np.random.random() * 2 * np.pi
            dx = radius * np.cos(theta)
            dy = radius * np.sin(theta)
            q[HelloStretchIdx.BASE_X] = x + dx
            q[HelloStretchIdx.BASE_Y] = y + dy
        return q

    def config_open_gripper(self, q):
        q[HelloStretchIdx.GRIPPER] = self.range[HelloStretchIdx.GRIPPER][1]
        return q

    def config_close_gripper(self, q):
        q[HelloStretchIdx.GRIPPER] = self.range[HelloStretchIdx.GRIPPER][0]
        return q

    def _update_joints(self, verbose: bool = False):
        """Get joint info from URDF or otherwise provide it"""
        self.joint_idx = [-1] * self.dof
        # Get the joint info we need from this
        joint_lift = self.ref.get_joint_info_by_name("joint_lift")
        self.range[:3, 0] = -float("Inf") * np.ones(3)
        self.range[:3, 1] = float("Inf") * np.ones(3)
        self.range[HelloStretchIdx.LIFT] = np.array(
            [
                joint_lift.lower_limit + self.joint_tol,
                joint_lift.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.LIFT] = joint_lift.index
        joint_head_pan = self.ref.get_joint_info_by_name("joint_head_pan")
        self.range[HelloStretchIdx.HEAD_PAN] = np.array(
            [
                joint_head_pan.lower_limit + self.joint_tol,
                joint_head_pan.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.HEAD_PAN] = joint_head_pan.index
        joint_head_tilt = self.ref.get_joint_info_by_name("joint_head_tilt")
        self.range[HelloStretchIdx.HEAD_TILT] = np.array(
            [joint_head_tilt.lower_limit, joint_head_tilt.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.HEAD_TILT] = joint_head_tilt.index
        joint_wrist_yaw = self.ref.get_joint_info_by_name("joint_wrist_yaw")
        self.range[HelloStretchIdx.WRIST_YAW] = np.array(
            [
                joint_wrist_yaw.lower_limit + self.joint_tol,
                joint_wrist_yaw.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.WRIST_YAW] = joint_wrist_yaw.index
        joint_wrist_roll = self.ref.get_joint_info_by_name("joint_wrist_roll")
        self.range[HelloStretchIdx.WRIST_ROLL] = np.array(
            [
                joint_wrist_roll.lower_limit + self.joint_tol,
                joint_wrist_roll.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.WRIST_ROLL] = joint_wrist_roll.index
        joint_wrist_pitch = self.ref.get_joint_info_by_name("joint_wrist_pitch")
        self.range[HelloStretchIdx.WRIST_PITCH] = np.array(
            [
                joint_wrist_pitch.lower_limit + self.joint_tol,
                joint_wrist_pitch.upper_limit - self.joint_tol,
            ]
        )
        self.joint_idx[HelloStretchIdx.WRIST_PITCH] = joint_wrist_pitch.index

        # arm position
        # TODO: fix this so that it is not hard-coded any more
        self.range[HelloStretchIdx.ARM] = np.array([0.0, 0.75])
        self.arm_idx = []
        upper_limit = 0
        for i in range(4):
            joint = self.ref.get_joint_info_by_name("joint_arm_l%d" % i)
            self.arm_idx.append(joint.index)
            upper_limit += joint.upper_limit

        # TODO: gripper
        self.gripper_idx = []
        for side in ["right", "left"]:
            joint = self.ref.get_joint_info_by_name("joint_gripper_finger_%s" % side)
            self.gripper_idx.append(joint.index)
            if verbose:
                print(side, joint.name, joint.lower_limit, joint.upper_limit)
            self.range[HelloStretchIdx.GRIPPER] = (
                np.array([joint.lower_limit, joint.upper_limit]) * 0.5
            )

        self._mins = self.range[:, 0]
        self._maxs = self.range[:, 1]
        self._rngs = self.range[:, 1] - self.range[:, 0]

    def get_backend(self):
        return self.backend

    def _set_joint_group(self, idxs, val):
        for idx in idxs:
            self.ref.set_joint_position(idx, val)

    def set_config(self, q):
        assert len(q) == self.dof
        x, y, theta = q[:3]
        # quaternion = pb.getQuaternionFromEuler((0, 0, theta))
        self.ref.set_pose((0, 0, self.base_height), [0, 0, 0, 1])
        # self.ref.set_pose((x, y, self.base_height), quaternion)
        self.ref.set_joint_position(0, x)
        self.ref.set_joint_position(1, y)
        self.ref.set_joint_position(2, theta)
        for idx, qq in zip(self.joint_idx, q):
            if idx < 0:
                continue
            self.ref.set_joint_position(idx, qq)
        # Finally set the arm and gripper as groups
        self._set_joint_group(self.arm_idx, q[HelloStretchIdx.ARM] / 4.0)
        self._set_joint_group(self.gripper_idx, q[HelloStretchIdx.GRIPPER])

    def interpolate(self, q0, qg, step=None, xy_tol=0.05, theta_tol=0.01):
        """interpolate from initial to final configuration. for this robot we break it up into
        four stages:
        1) rotate to point towards final location
        2) drive to final location
        3) rotate to final orientation
        4) move everything else
        """
        if step is None:
            step = self.default_step
        qi = q0.copy()
        theta0 = q0[HelloStretchIdx.BASE_THETA]
        thetag = qg[HelloStretchIdx.BASE_THETA]
        xy0 = q0[[HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_Y]]
        xyg = qg[[HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_Y]]
        dist = np.linalg.norm(xy0 - xyg)
        if dist > xy_tol:
            dx, dy = xyg - xy0
            theta = np.arctan2(dy, dx)
            for qi, ai in self.interpolate_angle(
                qi, theta0, theta, step[HelloStretchIdx.BASE_THETA]
            ):
                yield qi, ai
            for qi, ai in self.interpolate_xy(qi, xy0, dist, step[HelloStretchIdx.BASE_X]):
                yield qi, ai
        else:
            theta = theta0
        # update angle
        if np.abs(thetag - theta) > theta_tol:
            for qi, ai in self.interpolate_angle(
                qi, theta, thetag, step[HelloStretchIdx.BASE_THETA]
            ):
                yield qi, ai
        # Finally interpolate the whole joint space
        for qi, ai in self.interpolate_arm(qi, qg, step):
            yield qi, ai

    def manip_fk(self, q: np.ndarray = None, node: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """manipulator specific forward kinematics; uses separate URDF than the full-body fk() method"""
        assert q.shape == (self.dof,)

        if "pinocchio" in self._ik_type:
            q = self._ros_pose_to_pinocchio(q)

        ee_pos, ee_quat = self.manip_ik_solver.compute_fk(q, node)
        return ee_pos.copy(), ee_quat.copy()

    def update_head(self, qi: np.ndarray, look_at) -> np.ndarray:
        """move head based on look_at and return the joint-state"""
        qi[HelloStretchIdx.HEAD_PAN] = look_at[0]
        qi[HelloStretchIdx.HEAD_TILT] = look_at[1]
        return qi

    def update_gripper(self, qi, open=True):
        """update target state for gripper"""
        if open:
            qi[HelloStretchIdx.GRIPPER] = self.GRIPPER_OPEN
        else:
            qi[HelloStretchIdx.GRIPPER] = self.GRIPPER_CLOSED
        return qi

    def interpolate_xy(self, qi, xy0, dist, step=0.1):
        """just move forward with step to target distance"""
        # create a trajectory here
        x, y = xy0
        theta = qi[HelloStretchIdx.BASE_THETA]
        while dist > 0:
            qi = self.update_head(qi.copy(), self.look_front)
            ai = np.zeros(self.dof)
            if dist > step:
                dx = step
            else:
                dx = dist
            dist -= dx
            x += np.cos(theta) * dx
            y += np.sin(theta) * dx
            # x += np.sin(theta) * dx
            # y += np.cos(theta) * dx
            qi[HelloStretchIdx.BASE_X] = x
            qi[HelloStretchIdx.BASE_Y] = y
            ai[0] = dx
            yield qi, ai

    def _to_ik_format(self, q):
        qi = np.zeros(self.ik_solver.get_num_joints())
        qi[0] = q[HelloStretchIdx.BASE_X]
        qi[1] = q[HelloStretchIdx.BASE_Y]
        qi[2] = q[HelloStretchIdx.BASE_THETA]
        qi[3] = q[HelloStretchIdx.LIFT]
        # Next 4 are all arm joints
        arm_ext = q[HelloStretchIdx.ARM] / 4.0
        qi[4] = arm_ext
        qi[5] = arm_ext
        qi[6] = arm_ext
        qi[7] = arm_ext
        # Wrist joints
        qi[8] = q[HelloStretchIdx.WRIST_YAW]
        qi[9] = q[HelloStretchIdx.WRIST_PITCH]
        qi[10] = q[HelloStretchIdx.WRIST_ROLL]
        return qi

    def _to_manip_format(self, q):
        qi = np.zeros(self._manip_dof)
        qi[0] = q[HelloStretchIdx.BASE_X]
        qi[1] = q[HelloStretchIdx.LIFT]
        # Next 4 are all arm joints
        arm_ext = q[HelloStretchIdx.ARM] / 4.0
        qi[2] = arm_ext
        qi[3] = arm_ext
        qi[4] = arm_ext
        qi[5] = arm_ext
        # Wrist joints
        qi[6] = q[HelloStretchIdx.WRIST_YAW]
        qi[7] = q[HelloStretchIdx.WRIST_PITCH]
        qi[8] = q[HelloStretchIdx.WRIST_ROLL]
        return qi

    def _to_plan_format(self, q):
        qi = np.zeros(self.dof)
        qi[HelloStretchIdx.BASE_X] = q[0]
        qi[HelloStretchIdx.BASE_Y] = q[1]
        qi[HelloStretchIdx.BASE_THETA] = q[2]
        qi[HelloStretchIdx.LIFT] = q[3]
        # Arm is sum of the next four joints
        qi[HelloStretchIdx.ARM] = q[4] + q[5] + q[6] + q[7]
        qi[HelloStretchIdx.WRIST_YAW] = q[8]
        qi[HelloStretchIdx.WRIST_PITCH] = q[9]
        qi[HelloStretchIdx.WRIST_ROLL] = q[10]
        return qi

    def _from_manip_format(self, q_raw, q_init):
        # combine arm telescoping joints
        # This is sort of an action representation
        # Compute the actual robot conmfiguration
        q = q_init.copy()
        # Get the theta - we can then convert this over to see where the robot will end up
        q[HelloStretchIdx.BASE_X] = q_raw[0]
        # q[HelloStretchIdx.BASE_Y] += 0
        # No change to theta
        q[HelloStretchIdx.LIFT] = q_raw[1]
        q[HelloStretchIdx.ARM] = np.sum(q_raw[2:6])
        q[HelloStretchIdx.WRIST_ROLL] = q_raw[8]
        q[HelloStretchIdx.WRIST_PITCH] = q_raw[7]
        q[HelloStretchIdx.WRIST_YAW] = q_raw[6]
        return q

    def _pinocchio_pose_to_ros(self, joint_angles):
        raise NotImplementedError

    def _ros_pose_to_pinocchio(self, joint_angles):
        """utility to convert Stretch joint angle output to pinocchio joint pose format"""
        pin_compatible_joints = np.zeros(9)
        pin_compatible_joints[0] = joint_angles[HelloStretchIdx.BASE_X]
        pin_compatible_joints[1] = joint_angles[HelloStretchIdx.LIFT]
        pin_compatible_joints[2] = pin_compatible_joints[3] = pin_compatible_joints[
            4
        ] = pin_compatible_joints[5] = (joint_angles[HelloStretchIdx.ARM] / 4)
        pin_compatible_joints[6] = joint_angles[HelloStretchIdx.WRIST_YAW]
        pin_compatible_joints[7] = joint_angles[HelloStretchIdx.WRIST_PITCH]
        pin_compatible_joints[8] = joint_angles[HelloStretchIdx.WRIST_ROLL]
        return pin_compatible_joints

    def ik(self, pose, q0):
        pos, rot = pose
        se3 = Rotation.from_quat(rot).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = np.array(se3).reshape(3, 3)
        x, y, z = pos
        pose[:3, 3] = np.array([x, y, z - self.base_height])
        q, success, debug_info = self.ik_solver.compute_ik(pose, self._to_ik_format(q0))
        if q is not None and success:
            return self._to_plan_format(q)
        else:
            return None

    def manip_ik(
        self,
        pose_query,
        q0=None,
        relative: bool = True,
        update_pb: bool = True,
        num_attempts: int = 1,
        verbose: bool = False,
        custom_ee_frame: Optional[str] = None,
    ):
        """IK in manipulation mode. Takes in a 4x4 pose_query matrix in se(3) and initial
        configuration of the robot.

        By default move relative. easier that way.
        """

        if q0 is not None:
            self._to_manip_format(q0)
            default_q = q0
        else:
            # q0 = STRETCH_HOME_Q
            default_q = STRETCH_HOME_Q
        # Perform IK
        # These should be relative to the robot's base
        if relative:
            pos, quat = pose_query
        else:
            # We need to compute this relative to the robot...
            # So how do we do that?
            # This logic currently in local hello robot client
            raise NotImplementedError()

        q, success, debug_info = self.manip_ik_solver.compute_ik(
            pos,
            quat,
            q0,
            num_attempts=num_attempts,
            verbose=verbose,
            custom_ee_frame=custom_ee_frame,
        )

        if q is not None and success:
            q = self._from_manip_format(q, default_q)
            # self.set_config(q)

        return q, success, debug_info

    def get_ee_pose(self, q=None):
        raise NotImplementedError()

    def interpolate_angle(self, qi, theta0, thetag, step=0.1):
        """just rotate to target angle"""
        if theta0 > thetag:
            thetag2 = thetag + 2 * np.pi
        else:
            thetag2 = thetag - 2 * np.pi
        dist1 = np.abs(thetag - theta0)
        dist2 = np.abs(thetag2 - theta0)
        # TODO remove debug code
        # print("....", qi)
        print("interp from", theta0, "to", thetag, "or maybe", thetag2)
        # print("dists =", dist1, dist2)
        if dist2 < dist1:
            dist = dist2
            thetag = thetag2
        else:
            dist = dist1
        # Dumb check to see if we can get the rotation right
        # if dist > np.pi:
        #    thetag -= np.pi/2
        #    dist = np.abs(thetag - theta0)
        # TODO remove debug code
        # print(" > interp from", theta0, "to", thetag)
        # Getting the direction right here
        dirn = 1.0 if thetag > theta0 else -1.0
        while dist > 0:
            qi = qi.copy()
            ai = np.zeros(self.dof)
            # TODO: we should handle look differently
            # qi = self.update_head(qi, self.look_front)
            if dist > step:
                dtheta = step
            else:
                dtheta = dist
            dist -= dtheta
            ai[2] = dirn * dtheta
            qi[HelloStretchIdx.BASE_THETA] += dirn * dtheta
            yield qi, ai

    def interpolate_arm(self, q0, qg, step=None):
        if step is None:
            step = self.default_step
        qi = q0
        while np.any(np.abs(qi - qg) > self.default_tols):
            qi = qi.copy()
            ai = qi.copy()
            ai[:3] = np.zeros(3)  # action does not move the base
            # TODO: we should handle look differently
            qi = self.update_head(qi.copy(), self.look_at_ee)
            dq = qg - qi
            dq = np.clip(dq, -1 * step, step)
            qi += dq
            qi = self.update_head(qi, self.look_at_ee)
            yield qi, ai

    def extend_arm_to(self, q, arm):
        """
        Extend the arm by a certain amount.
        Move the base as well to compensate.

        This is purely a helper function to make sure that we can find poses at which we can
        extend the arm in order to grasp.
        """
        a0 = q[HelloStretchIdx.ARM]
        a1 = arm
        q = q.copy()
        q[HelloStretchIdx.ARM] = a1
        theta = q[HelloStretchIdx.BASE_THETA] + np.pi / 2
        da = a1 - a0
        dx, dy = da * np.cos(theta), da * np.sin(theta)
        q[HelloStretchIdx.BASE_X] += dx
        q[HelloStretchIdx.BASE_Y] += dy
        return q

    def create_action_from_config(self, q: np.ndarray) -> ContinuousFullBodyAction:
        """Create a default interface action from this"""
        xyt = np.zeros(3)
        xyt[0] = q[HelloStretchIdx.BASE_X]
        xyt[1] = q[HelloStretchIdx.BASE_Y]
        xyt[2] = q[HelloStretchIdx.BASE_THETA]
        return self.create_action(
            lift=q[HelloStretchIdx.LIFT],
            arm=q[HelloStretchIdx.ARM],
            pitch=q[HelloStretchIdx.WRIST_PITCH],
            roll=q[HelloStretchIdx.WRIST_ROLL],
            yaw=q[HelloStretchIdx.WRIST_YAW],
            xyt=xyt,
        )

    def create_action(
        self,
        lift=None,
        arm=None,
        roll=None,
        pitch=None,
        yaw=None,
        pan=None,
        tilt=None,
        xyt=None,
        defaults: np.ndarray = None,
    ) -> ContinuousFullBodyAction:
        """
        Original Arm Action Space: We define the action space that jointly controls (1) arm extension (horizontal), (2) arm height (vertical), (3) gripper wrist’s roll, pitch, and yaw, and (4) the camera’s yaw and pitch. The resulting size of the action space is 10.
        - Arm extension (size: 4): It consists of 4 motors that extend the arm: joint_arm_l0 (index 28 in robot interface), joint_arm_l1 (27), joint_arm_l2 (26), joint_arm_l3 (25)
        - Arm height (size: 1): It consists of 1 motor that moves the arm vertically: joint_lift (23)
        - Gripper wrist (size: 3): It consists of 3 motors that control the roll, pitch, and yaw of the gripper wrist: joint_wrist_yaw (31),  joint_wrist_pitch (39),  joint_wrist_roll (40)
        - Camera (size 2): It consists of 2 motors that control the yaw and pitch of the camera: joint_head_pan (7), joint_head_tilt (8)

        As a result, the original action space is the order of [joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3, joint_lift, joint_wrist_yaw, joint_wrist_pitch, joint_wrist_roll, joint_head_pan, joint_head_tilt] defined in habitat/robots/stretch_robot.py
        """
        assert self.joints_dof == 10
        if defaults is None:
            joints = np.zeros(self.joints_dof)
        else:
            assert len(defaults) == self.joints_dof
            joints = defaults.copy()
        if arm is not None:
            joints[:4] = np.ones(4) * (arm / 4.0)
        if lift is not None:
            joints[4] = lift
        if roll is not None:
            joints[5] = roll
        if pitch is not None:
            joints[6] = pitch
        if yaw is not None:
            joints[7] = yaw
        if pan is not None:
            joints[8] = pan
        if tilt is not None:
            joints[9] = tilt
        return ContinuousFullBodyAction(joints=joints, xyt=xyt)

    def manip_ik_for_grasp_frame(self, ee_pos, ee_rot, q0: Optional[np.ndarray] = None) -> Tuple:
        # Construct the final end effector pose
        if ee_pos[1] > 0:
            raise RuntimeError(
                f"{self.name}: graspable objects should be in the negative y direction, got this target position: {ee_pos}"
            )
        elif ee_pos[2] < 0:
            raise RuntimeError(
                f"{self.name}: graspable objects should be above the ground, got this target position: {ee_pos}"
            )

        if len(q0) != self._manip_dof:
            assert (
                len(q0) == self.dof
            ), f"Joint states size must be either full = {self.dof} or manipulator = {self._manip_dof} dof"
            q0 = self._to_manip_format(q0)

        target_joint_state, success, info = self.manip_ik((ee_pos, ee_rot), q0=q0)
        return target_joint_state, ee_pos, ee_rot, success, info


if __name__ == "__main__":
    robot = HelloStretchKinematics()
    q0 = STRETCH_HOME_Q.copy()
    q1 = STRETCH_HOME_Q.copy()
    q0[2] = -1.18
    q1[2] = -1.1
    for state, action in robot.interpolate_angle(q0, q0[2], q1[2]):
        print(action)
