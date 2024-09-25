# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

import numpy as np
import pinocchio as pin

# from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R

from stretch.dynav.ok_robot_hw.global_parameters import *
from stretch.dynav.ok_robot_hw.utils import transform_joint_array
from stretch.motion.kinematics import HelloStretchIdx

OVERRIDE_STATES: dict[str, float] = {}


class HelloRobot:
    def __init__(
        self,
        robot,
        stretch_client_urdf_file="src/stretch/config/urdf",
        gripper_threshold=7.0,
        stretch_gripper_max=0.64,
        stretch_gripper_min=0,
        end_link="link_gripper_s3_body",
    ):
        self.STRETCH_GRIPPER_MAX = stretch_gripper_max
        self.STRETCH_GRIPPER_MIN = stretch_gripper_min
        self.urdf_path = os.path.join(stretch_client_urdf_file, "stretch.urdf")
        self.joints_pin = {"joint_fake": 0}

        self.GRIPPER_THRESHOLD = gripper_threshold

        print("hello robot starting")
        self.head_joint_list = ["joint_fake", "joint_head_pan", "joint_head_tilt"]
        self.init_joint_list = [
            "joint_fake",
            "joint_lift",
            "3",
            "2",
            "1",
            "0",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_gripper_finger_left",
        ]

        # end_link is the frame of reference node
        self.end_link = end_link
        self.set_end_link(end_link)

        # Initialize StretchClient controller
        self.robot = robot
        self.robot.switch_to_manipulation_mode()
        # time.sleep(2)

        # Constraining the robots movement
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        self.pan, self.tilt = self.robot.get_pan_tilt()

    def set_end_link(self, link):
        if link == GRIPPER_FINGERTIP_LEFT_NODE or GRIPPER_FINGERTIP_RIGHT_NODE:
            self.joint_list = self.init_joint_list
        else:
            self.joint_list = self.init_joint_list[:-1]

    def get_joints(self):
        """
        Returns all the joint names and values involved in forward kinematics of head and gripper
        """
        ## Names of all 13 joints
        joint_names = (
            self.init_joint_list + ["joint_gripper_finger_right"] + self.head_joint_list[1:]
        )
        self.updateJoints()
        joint_values = list(self.joints.values()) + [0] + list(self.head_joints.values())[1:]

        return joint_names, joint_values

    def move_to_position(
        self,
        lift_pos=None,
        arm_pos=None,
        base_trans=0.0,
        wrist_yaw=None,
        wrist_pitch=None,
        wrist_roll=None,
        gripper_pos=None,
        base_theta=None,
        head_tilt=None,
        head_pan=None,
        blocking=True,
    ):
        """
        Moves the robots, base, arm, lift, wrist and head to a desired position.
        """
        if base_theta is not None:
            self.robot.navigate_to([0, 0, base_theta], relative=True, blocking=True)
            return

        # Base, arm and lift state update
        target_state = self.robot.get_six_joints()
        if not gripper_pos is None:
            self.CURRENT_STATE = (
                gripper_pos * (self.STRETCH_GRIPPER_MAX - self.STRETCH_GRIPPER_MIN)
                + self.STRETCH_GRIPPER_MIN
            )
            self.robot.gripper_to(self.CURRENT_STATE, blocking=blocking)
        if not arm_pos is None:
            target_state[2] = arm_pos
        if not lift_pos is None:
            target_state[1] = lift_pos
        if base_trans is None:
            base_trans = 0
        target_state[0] = base_trans + target_state[0]

        # Wrist state update
        if not wrist_yaw is None:
            target_state[3] = wrist_yaw
        if not wrist_pitch is None:
            target_state[4] = min(wrist_pitch, 0.1)
        if not wrist_roll is None:
            target_state[5] = wrist_roll

        # Actual Movement
        print("Expected", target_state)
        print("Actual", self.robot.get_six_joints())
        print("Error", target_state - self.robot.get_six_joints())
        # print('Target Position', target_state)
        # print('pan tilt before', self.robot.get_pan_tilt())
        self.robot.arm_to(target_state, blocking=blocking, head=np.array([self.pan, self.tilt]))
        # print('pan tilt after', self.robot.get_pan_tilt())
        # print('Actual location', self.robot.get_six_joints())

        # Head state update and Movement
        # target_head_pan, target_head_tilt = self.robot.get_pan_tilt()
        target_head_pan = self.pan
        target_head_tilt = self.tilt
        if not head_tilt is None:
            target_head_tilt = head_tilt
            self.tilt = head_tilt
        if not head_pan is None:
            target_head_pan = head_pan
            self.pan = head_pan
        self.robot.head_to(head_tilt=target_head_tilt, head_pan=target_head_pan, blocking=blocking)
        # self.pan, self.tilt = self.robot.get_pan_tilt()
        # time.sleep(0.7)

    def pickup(self, width):
        """
        Code for grasping the object
        Gripper closes gradually until it encounters resistance
        """
        next_gripper_pos = width
        while True:
            self.robot.gripper_to(
                max(next_gripper_pos * self.STRETCH_GRIPPER_MAX, -0.2), blocking=True
            )
            curr_gripper_pose = self.robot.get_gripper_position()
            # print('Robot means to move gripper to', next_gripper_pos * self.STRETCH_GRIPPER_MAX)
            # print('Robot actually moves gripper to', curr_gripper_pose)
            if next_gripper_pos == -1:
                break

            if next_gripper_pos > 0:
                next_gripper_pos -= 0.35
            else:
                next_gripper_pos = -1

    def updateJoints(self):
        """
        update all the current positions of joints
        """
        state = self.robot.get_six_joints()
        origin_dist = state[0]

        # Head Joints
        pan, tilt = self.robot.get_pan_tilt()

        self.joints_pin["joint_fake"] = origin_dist
        self.joints_pin["joint_lift"] = state[1]
        armPos = state[2]
        self.joints_pin["joint_arm_l3"] = armPos / 4.0
        self.joints_pin["joint_arm_l2"] = armPos / 4.0
        self.joints_pin["joint_arm_l1"] = armPos / 4.0
        self.joints_pin["joint_arm_l0"] = armPos / 4.0
        self.joints_pin["joint_wrist_yaw"] = state[3]
        self.joints_pin["joint_wrist_roll"] = state[5]
        self.joints_pin["joint_wrist_pitch"] = OVERRIDE_STATES.get("wrist_pitch", state[4])
        self.joints_pin["joint_gripper_finger_left"] = 0

        self.joints_pin["joint_head_pan"] = pan
        self.joints_pin["joint_head_tilt"] = tilt

    # following function is used to move the robot to a desired joint configuration
    def move_to_joints(self, joints, gripper, mode=0):
        """
        Given the desired joints movement this function will the joints accordingly
        """
        state = self.robot.get_six_joints()

        # clamp rotational joints between -1.57 to 1.57
        joints["joint_wrist_pitch"] = (joints["joint_wrist_pitch"] + 1.57) % 3.14 - 1.57
        joints["joint_wrist_yaw"] = (joints["joint_wrist_yaw"] + 1.57) % 3.14 - 1.57
        joints["joint_wrist_roll"] = (joints["joint_wrist_roll"] + 1.57) % 3.14 - 1.57
        joints["joint_wrist_pitch"] = self.clamp(joints["joint_wrist_pitch"], -1.57, 0.1)
        target_state = [
            joints["joint_fake"],
            joints["joint_lift"],
            joints["3"] + joints["2"] + joints["1"] + joints["0"],
            joints["joint_wrist_yaw"],
            joints["joint_wrist_pitch"],
            joints["joint_wrist_roll"],
        ]

        # print('pan tilt before', self.robot.get_pan_tilt())

        # Moving only the lift first
        if mode == 1:
            target1 = state
            target1[0] = target_state[0]
            target1[1] = min(1.1, target_state[1] + 0.2)
            self.robot.arm_to(target1, blocking=True, head=np.array([self.pan, self.tilt]))

        self.robot.arm_to(target_state, blocking=True, head=np.array([self.pan, self.tilt]))
        self.robot.head_to(head_tilt=self.tilt, head_pan=self.pan, blocking=True)

        self.robot.arm_to(target_state, blocking=True, head=np.array([self.pan, self.tilt]))
        self.robot.head_to(head_tilt=self.tilt, head_pan=self.pan, blocking=True)

        # print('pan tilt after', self.robot.get_pan_tilt())
        # print(f"current state {self.robot.get_six_joints()}")
        # print(f"target state {target_state}")
        # time.sleep(1)

        # NOTE: below code is to fix the pitch drift issue in current hello-robot. Remove it if there is no pitch drift issue
        OVERRIDE_STATES["wrist_pitch"] = joints["joint_wrist_pitch"]

    def get_joint_transform(self, node1, node2):
        """
        This function takes two nodes from a robot URDF file as input and
        outputs the coordinate frame of node2 relative to the coordinate frame of node1.

        Mainly used for transforming coordinates from camera frame to gripper frame.
        """

        # return frame_transform, frame2, frame1
        self.updateJoints()
        frame_pin = self.robot.get_frame_pose(self.joints_pin, node1, node2)

        return frame_pin

    def move_to_pose(self, translation_tensor, rotational_tensor, gripper, move_mode=0):
        """
        Function to move the gripper to a desired translation and rotation
        """
        translation = [translation_tensor[0], translation_tensor[1], translation_tensor[2]]
        rotation = rotational_tensor

        self.updateJoints()

        q = self.robot.get_joint_positions()
        q[HelloStretchIdx.WRIST_PITCH] = OVERRIDE_STATES.get(
            "wrist_pitch", q[HelloStretchIdx.WRIST_PITCH]
        )
        pin_pose = self.robot.get_ee_pose(matrix=True, link_name=self.end_link, q=q)
        pin_rotation, pin_translation = pin_pose[:3, :3], pin_pose[:3, 3]
        pin_curr_pose = pin.SE3(pin_rotation, pin_translation)

        rot_matrix = R.from_euler("xyz", rotation, degrees=False).as_matrix()

        pin_del_pose = pin.SE3(np.array(rot_matrix), np.array(translation))
        pin_goal_pose_new = pin_curr_pose * pin_del_pose

        final_pos = pin_goal_pose_new.translation.tolist()
        final_quat = pin.Quaternion(pin_goal_pose_new.rotation).coeffs().tolist()
        # print(f"final pos and quat {final_pos}\n {final_quat}")

        full_body_cfg = self.robot.solve_ik(
            final_pos, final_quat, None, False, node_name=self.end_link
        )
        if full_body_cfg is None:
            print("Warning: Cannot find an IK solution for desired EE pose!")
            return False

        pin_joint_pos = self.robot._extract_joint_pos(full_body_cfg)
        transform_joint_pos = transform_joint_array(pin_joint_pos)

        self.joint_array1 = transform_joint_pos

        ik_joints = {}
        for joint_index in range(len(self.joint_array1)):
            ik_joints[self.joint_list[joint_index]] = self.joint_array1[joint_index]

        # Actual Movement of joints
        self.move_to_joints(ik_joints, gripper, move_mode)
