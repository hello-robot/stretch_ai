# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import numpy as np
import pinocchio as pin

from stretch.dynav.ok_robot_hw.global_parameters import *
from stretch.dynav.ok_robot_hw.image_publisher import ImagePublisher
from stretch.dynav.ok_robot_hw.utils.utils import apply_se3_transform


def capture_and_process_image(camera, mode, obj, socket, hello_robot):

    print("Currently in " + mode + " mode and the robot is about to manipulate " + obj + ".")

    image_publisher = ImagePublisher(camera, socket)

    # Centering the object
    head_tilt_angles = [0, -0.1, 0.1]
    tilt_retries, side_retries = 1, 0
    retry_flag = True
    head_tilt = INIT_HEAD_TILT
    head_pan = INIT_HEAD_PAN

    while retry_flag:
        translation, rotation, depth, width, retry_flag = image_publisher.publish_image(
            obj, mode, head_tilt=head_tilt
        )

        print(f"retry flag : {retry_flag}")
        if retry_flag == 1:
            base_trans = translation[0]
            head_tilt += rotation[0]

            hello_robot.move_to_position(
                base_trans=base_trans, head_pan=head_pan, head_tilt=head_tilt
            )
            time.sleep(1)

        elif retry_flag != 0 and side_retries == 3:
            print("Tried in all angles but couldn't succeed")
            time.sleep(1)
            return None, None, None, None

        elif side_retries == 2 and tilt_retries == 3:
            hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
            side_retries = 3

        elif retry_flag == 2:
            if tilt_retries == 3:
                if side_retries == 0:
                    hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
                    side_retries = 1
                else:
                    hello_robot.move_to_position(base_trans=-0.2, head_tilt=head_tilt)
                    side_retries = 2
                tilt_retries = 1
            else:
                print(f"retrying with head tilt : {head_tilt + head_tilt_angles[tilt_retries]}")
                hello_robot.move_to_position(
                    head_pan=head_pan, head_tilt=head_tilt + head_tilt_angles[tilt_retries]
                )
                tilt_retries += 1
                time.sleep(1)

    if mode == "place":
        translation = np.array([-translation[1], -translation[0], -translation[2]])

    if mode == "pick":
        return rotation, translation, depth, width
    else:
        return rotation, translation


def move_to_point(robot, point, base_node, gripper_node, move_mode=1, pitch_rotation=0):
    """
    Function for moving the gripper to a specific point
    """
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    dest_frame = pin.SE3(rotation, point)
    transform = robot.get_joint_transform(base_node, gripper_node)

    # Rotation from gripper frame frame to gripper frame
    transformed_frame = transform * dest_frame

    transformed_frame.translation[2] -= 0.2

    robot.move_to_pose(
        [
            transformed_frame.translation[0],
            transformed_frame.translation[1],
            transformed_frame.translation[2],
        ],
        [pitch_rotation, 0, 0],
        [1],
        move_mode=move_mode,
    )


def pickup(
    robot,
    rotation,
    translation,
    base_node,
    gripper_node,
    gripper_height=0.03,
    gripper_depth=0.03,
    gripper_width=1,
):
    """
    rotation: Relative rotation of gripper pose w.r.t camera
    translation: Relative translation of gripper pose w.r.t camera
    base_node: Camera Node

    Supports home robot top down grasping as well

    Graping trajectory steps
    1. Rotation of gripper
    2. Lift the gripper
    3. Move the base such gripper in line with the grasp
    4. Gradually Move the gripper to the desired position
    """
    # Transforming the final point from Model camera frame to robot camera frame
    pin_point = np.array([-translation[1], -translation[0], translation[2]])

    # Rotation from Camera frame to Model frame
    rotation1_bottom_mat = np.array(
        [
            [0.0000000, -1.0000000, 0.0000000],
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
        ]
    )

    # Rotation from model frame to pose frame
    rotation1_mat = np.array(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ]
    )

    # Rotation from camera frame to pose frame
    pin_rotation = np.dot(rotation1_bottom_mat, rotation1_mat)
    # print(f"pin rotation{pin_rotation}")

    # Relative rotation and translation of grasping point relative to camera
    pin_dest_frame = pin.SE3(np.array(pin_rotation), np.array(pin_point))
    # print(f"pin dest frame {pin_dest_frame}")

    # Camera to gripper frame transformation
    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)

    # transformed_frame = del_pose * dest_frame
    pin_transformed_frame = pin_cam2gripper_transform * pin_dest_frame
    # print(f"pin_transformed frame {pin_transformed_frame}")

    # Lifting the arm to high position as part of pregrasping position
    print("pan, tilt before", robot.robot.get_pan_tilt())
    robot.move_to_position(gripper_pos=gripper_width)
    robot.move_to_position(lift_pos=1.05, head_pan=None, head_tilt=None)
    print("pan, tilt after", robot.robot.get_pan_tilt())

    # Rotation for aligning Robot gripper frame to Model gripper frame
    rotation2_top_mat = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])

    # final Rotation of gripper to hold the objcet
    pin_final_rotation = np.dot(pin_transformed_frame.rotation, rotation2_top_mat)
    print(f"pin final rotation {pin_final_rotation}")

    rpy_angles = pin.rpy.matrixToRpy(pin_final_rotation)
    print("pan, tilt before", robot.robot.get_pan_tilt())
    robot.move_to_pose(
        [0, 0, 0],
        [rpy_angles[0], rpy_angles[1], rpy_angles[2]],
        [1],
    )
    print("pan, tilt after", robot.robot.get_pan_tilt())

    # Final grasping point relative to camera
    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    pin_transformed_point1 = apply_se3_transform(pin_cam2gripper_transform, pin_point)
    # print(f"pin transformed point1 {pin_transformed_point1}")

    # Final grasping point relative to base
    pin_cam2base_transform = robot.get_joint_transform(base_node, "base_link")
    pin_base_point = apply_se3_transform(pin_cam2base_transform, pin_point)
    # print(f"pin base point {pin_base_point}")

    diff_value = (
        0.225 - gripper_depth - gripper_height
    )  # 0.228 is the distance between link_Straight_gripper node and the gripper tip
    pin_transformed_point1[2] -= diff_value
    ref_diff = diff_value

    # Moving gripper to a point that is 0.2m away from the pose center in the line of gripper
    print("pan, tilt before", robot.robot.get_pan_tilt())
    robot.move_to_pose(
        [pin_transformed_point1[0], pin_transformed_point1[1], pin_transformed_point1[2] - 0.2],
        [0, 0, 0],
        [1],
        move_mode=1,
    )
    print("pan, tilt after", robot.robot.get_pan_tilt())

    # Z-Axis of link_straight_gripper points in line of gripper
    # So, the z co-ordiante of point w.r.t gripper gives the distance of point from gripper
    pin_base2gripper_transform = robot.get_joint_transform("base_link", gripper_node)
    pin_transformed_point2 = apply_se3_transform(pin_base2gripper_transform, pin_base_point)
    curr_diff = pin_transformed_point2[2]

    # The distance between gripper and point is covered gradullay to allow for velocity control when it approaches the object
    # Lower velocity helps is not topping the light objects
    diff = abs(curr_diff - ref_diff)
    if diff > 0.08:
        dist = diff - 0.08
        state = robot.robot.get_six_joints()
        state[1] += 0.02
        # state[0] -= 0.015
        robot.robot.arm_to(state, blocking=True)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist

    while diff > 0.01:
        dist = min(0.03, diff)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist

    # Now the gripper reached the grasping point and starts picking procedure
    robot.pickup(gripper_width)

    # Lifts the arm
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1] + 0.2, 1.1))

    # Tucks the gripper so that while moving to place it won't collide with any obstacles
    robot.move_to_position(arm_pos=0.01)
    robot.move_to_position(wrist_pitch=0.0)
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1], 0.9), wrist_yaw=2.5)
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1], 0.55))
