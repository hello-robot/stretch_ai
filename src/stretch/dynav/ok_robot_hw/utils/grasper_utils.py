import time
import math
import numpy as np
import pinocchio as pin

from stretch.dynav.ok_robot_hw.image_publisher import ImagePublisher
from stretch.dynav.ok_robot_hw.global_parameters import *
from stretch.dynav.ok_robot_hw.utils.utils import apply_se3_transform

def capture_and_process_image(camera, mode, obj, socket, hello_robot):
    
    print('Currently in ' + mode + ' mode and the robot is about to manipulate ' + obj + '.')

    image_publisher = ImagePublisher(camera, socket)

    # Centering the object
    head_tilt_angles = [0, -0.1, 0.1]
    tilt_retries, side_retries = 1, 0
    retry_flag = True
    head_tilt = INIT_HEAD_TILT
    head_pan = INIT_HEAD_PAN

    while(retry_flag):
        translation, rotation, depth, width, retry_flag = image_publisher.publish_image(obj, mode, head_tilt=head_tilt)

        print(f"retry flag : {retry_flag}")
        if (retry_flag == 1):
            base_trans = translation[0]
            head_tilt += (rotation[0])

            hello_robot.move_to_position(base_trans=base_trans,
                                    head_pan=head_pan,
                                    head_tilt=head_tilt)
            time.sleep(4)
        
        elif (retry_flag !=0 and side_retries == 3):
            print("Tried in all angles but couldn't succed")
            time.sleep(2)
            return None, None, None

        elif (side_retries == 2 and tilt_retries == 3):
            hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
            side_retries = 3

        elif retry_flag == 2:
            if (tilt_retries == 3):
                if (side_retries == 0):
                    hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
                    side_retries = 1
                else:
                    hello_robot.move_to_position(base_trans=-0.2, head_tilt=head_tilt)
                    side_retries = 2
                tilt_retries = 1
            else:
                print(f"retrying with head tilt : {head_tilt + head_tilt_angles[tilt_retries]}")
                hello_robot.move_to_position(head_pan=head_pan,
                                        head_tilt=head_tilt + head_tilt_angles[tilt_retries])
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
    # import PyKDL
    # rotation = PyKDL.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1)
    rotation = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    # dest_frame = PyKDL.Frame(rotation, point)
    dest_frame = pin.SE3(rotation, point)
    # transform, _, _ = robot.get_joint_transform(base_node, gripper_node)
    transform = robot.get_joint_transform(base_node, gripper_node)

    # Rotation from gripper frame frame to gripper frame
    transformed_frame = transform * dest_frame

    transformed_frame.translation[2] -= 0.2

    robot.move_to_pose(
            [transformed_frame.translation[0], transformed_frame.translation[1], transformed_frame.translation[2]],
            [pitch_rotation, 0, 0],
            [1],
            move_mode=move_mode
        )

def pickup(robot, rotation, translation, base_node, gripper_node, gripper_height = 0.03, gripper_depth=0.03, gripper_width = 1):
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
    import PyKDL
    # Transforming the final point from Model camera frame to robot camera frame
    point = PyKDL.Vector(-translation[1], -translation[0], translation[2])
    pin_point = np.array([-translation[1], -translation[0], translation[2]])

    # Rotation from Camera frame to Model frame
    rotation1_bottom = PyKDL.Rotation(0.0000000, -1.0000000,  0.0000000,
                                -1.0000000,  0.0000000,  0.0000000, 
                                0.0000000,  0.0000000, 1.0000000)
    rotation1_bottom_mat = np.array([[0.0000000, -1.0000000,  0.0000000],
                                 [-1.0000000,  0.0000000,  0.0000000],
                                 [0.0000000,  0.0000000, 1.0000000]])

    # Rotation from model frame to pose frame
    rotation1 = PyKDL.Rotation(rotation[0][0], rotation[0][1], rotation[0][2],
                            rotation[1][0],  rotation[1][1], rotation[1][2],
                                rotation[2][0],  rotation[2][1], rotation[2][2])
    rotation1_mat = np.array([[rotation[0][0], rotation[0][1], rotation[0][2]],
                            [rotation[1][0],  rotation[1][1], rotation[1][2]],
                            [rotation[2][0],  rotation[2][1], rotation[2][2]]])

    # Rotation from camera frame to pose frame
    rotation =  rotation1_bottom * rotation1
    pin_rotation = np.dot(rotation1_bottom_mat, rotation1_mat)
    print(f"rotation {rotation}")
    print(f"pin rotation{pin_rotation}")

    # Relative rotation and translation of grasping point relative to camera
    dest_frame = PyKDL.Frame(rotation, point) 
    pin_dest_frame = pin.SE3(np.array(pin_rotation), np.array(pin_point))
    print(f"dest frame {dest_frame}")
    print(f"pin dest frame {pin_dest_frame}")

    # Camera to gripper frame transformation
    # cam2gripper_transform, pin_cam2gripper_transform, _, _ = robot.get_joint_transform(base_node, gripper_node)
    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)

    del_pose = PyKDL.Frame()
    del_rot = PyKDL.Rotation(PyKDL.Vector(pin_cam2gripper_transform.rotation[0][0], pin_cam2gripper_transform.rotation[1][0], pin_cam2gripper_transform.rotation[2][0]),
                                PyKDL.Vector(pin_cam2gripper_transform.rotation[0][1], pin_cam2gripper_transform.rotation[1][1], pin_cam2gripper_transform.rotation[2][1]),
                                PyKDL.Vector(pin_cam2gripper_transform.rotation[0][2], pin_cam2gripper_transform.rotation[1][2], pin_cam2gripper_transform.rotation[2][2]))
    del_trans = PyKDL.Vector(pin_cam2gripper_transform.translation[0], pin_cam2gripper_transform.translation[1], pin_cam2gripper_transform.translation[2])
    del_pose.M = del_rot
    del_pose.p = del_trans

    transformed_frame = del_pose * dest_frame
    pin_transformed_frame = pin_cam2gripper_transform * pin_dest_frame
    print(f"transformed frame {transformed_frame}")
    print(f"pin_transformed frame {pin_transformed_frame}")


    # Lifting the arm to high position as part of pregrasping position
    robot.move_to_position(lift_pos = 1.05, gripper_pos = gripper_width, head_pan = None, head_tilt = None)
    # time.sleep(2)

    # Rotation for aligning Robot gripper frame to Model gripper frame
    rotation2_top = PyKDL.Rotation(0, 0, 1, 1, 0, 0, 0, -1, 0)
    rotation2_top_mat = np.array([[0, 0, 1], 
                                [1, 0, 0],
                                [0, -1, 0]])


    # final Rotation of gripper to hold the objet
    final_rotation = transformed_frame.M * rotation2_top
    pin_final_rotation = np.dot(pin_transformed_frame.rotation, rotation2_top_mat)
    print(f"final rotation - {final_rotation}")
    print(f"pin final rotation {pin_final_rotation}")

    print("Rotation wrist with sleep of 2s")
    # robot.move_to_pose(
    #         [0, 0, 0],
    #         [final_rotation.GetRPY()[0], final_rotation.GetRPY()[1], final_rotation.GetRPY()[2]],
    #         [1],
    #     )
    rpy_angles = pin.rpy.matrixToRpy(pin_final_rotation)
    print(f"rpy angles {final_rotation.GetRPY}")
    print(f"pin rpy angles {rpy_angles}")
    robot.move_to_position(gripper_pos = gripper_width)
    robot.move_to_pose(
            [0, 0, 0],
            [rpy_angles[0], rpy_angles[1], rpy_angles[2]],
            [1],
        )
    time.sleep(1)

    # Final grasping point relative to camera
    # cam2gripper_transform, pin_cam2gripper_transform, _, _ = robot.get_joint_transform(base_node, gripper_node)
    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    del_pose = PyKDL.Frame()
    del_rot = PyKDL.Rotation(PyKDL.Vector(pin_cam2gripper_transform.rotation[0][0], pin_cam2gripper_transform.rotation[1][0], pin_cam2gripper_transform.rotation[2][0]),
                                PyKDL.Vector(pin_cam2gripper_transform.rotation[0][1], pin_cam2gripper_transform.rotation[1][1], pin_cam2gripper_transform.rotation[2][1]),
                                PyKDL.Vector(pin_cam2gripper_transform.rotation[0][2], pin_cam2gripper_transform.rotation[1][2], pin_cam2gripper_transform.rotation[2][2]))
    del_trans = PyKDL.Vector(pin_cam2gripper_transform.translation[0], pin_cam2gripper_transform.translation[1], pin_cam2gripper_transform.translation[2])
    del_pose.M = del_rot
    del_pose.p = del_trans
    transformed_point1 = del_pose * point
    pin_transformed_point1 = apply_se3_transform(pin_cam2gripper_transform, pin_point)
    print(f"transformed point1 {transformed_point1}")
    print(f"pin transformed point1 {pin_transformed_point1}")

    # Final grasping point relative to base
    # cam2base_transform, pin_cam2base_transform, _, _ = robot.get_joint_transform(base_node, 'base_link')
    pin_cam2base_transform = robot.get_joint_transform(base_node, 'base_link')
    del_pose = PyKDL.Frame()
    del_rot = PyKDL.Rotation(PyKDL.Vector(pin_cam2base_transform.rotation[0][0], pin_cam2base_transform.rotation[1][0], pin_cam2base_transform.rotation[2][0]),
                                PyKDL.Vector(pin_cam2base_transform.rotation[0][1], pin_cam2base_transform.rotation[1][1], pin_cam2base_transform.rotation[2][1]),
                                PyKDL.Vector(pin_cam2base_transform.rotation[0][2], pin_cam2base_transform.rotation[1][2], pin_cam2base_transform.rotation[2][2]))
    del_trans = PyKDL.Vector(pin_cam2base_transform.translation[0], pin_cam2base_transform.translation[1], pin_cam2base_transform.translation[2])
    del_pose.M = del_rot
    del_pose.p = del_trans
    base_point = del_pose * point
    pin_base_point = apply_se3_transform(pin_cam2base_transform, pin_point)
    print(f"base point {base_point}")
    print(f"pin base point {pin_base_point}")

    diff_value = (0.228 - gripper_depth - gripper_height) # 0.228 is the distance between link_Straight_gripper node and the gripper tip
    pin_transformed_point1[2] -= (diff_value)
    ref_diff = (diff_value)

    # Moving gripper to a point that is 0.2m away from the pose center in the line of gripper
    print("Moving to 0.2m away point with 4s sleep")
    # robot.move_to_pose(
    #     [transformed_point1.x(), transformed_point1.y(), transformed_point1.z() - 0.2],
    #     [0, 0, 0],
    #     [1],
    #     move_mode = 1
    # )
    robot.move_to_pose(
        [pin_transformed_point1[0], pin_transformed_point1[1], pin_transformed_point1[2] - 0.2],
        [0, 0, 0],
        [1],
        move_mode = 1
    )
    # time.sleep(4)

    # Z-Axis of link_straight_gripper points in line of gripper
    # So, the z co-ordiante of point w.r.t gripper gives the distance of point from gripper
    # base2gripper_transform, pin_base2gripper_transform, _, _ = robot.get_joint_transform('base_link', gripper_node)
    pin_base2gripper_transform = robot.get_joint_transform('base_link', gripper_node)
    # transformed_point2 = base2gripper_transform * base_point
    pin_transformed_point2 = apply_se3_transform(pin_base2gripper_transform, pin_base_point)
    # print(f"transformed point2 : {transformed_point2}")
    print(f"pin transformed point2 : {pin_transformed_point2}")
    # curr_diff = transformed_point2.z()
    curr_diff = pin_transformed_point2[2]

    # The distance between gripper and point is covered gradullay to allow for velocity control when it approaches the object
    # Lower velocity helps is not topping the light objects
    diff = abs(curr_diff - ref_diff)
    velocities = [1.0]*8
    velocities[5:] = [0.03, 0.03, 0.03, 0.03]
    velocities[0] = 0.03
    if diff > 0.08:
        dist = diff - 0.08
        print("Move to intermediate point with sleep 2s")
        robot.move_to_pose(
            [0, 0, dist],
            [0, 0, 0],
            [1]
        )
        # time.sleep(2)
        # base2gripper_transform, _, _ = robot.get_joint_transform('base_link', gripper_node)
        # print(f"transformed point3 : {base2gripper_transform * base_point}")
        diff = diff - dist
        
    while diff > 0.01:
        dist = min(0.03, diff)
        print("Move to Secondary intermediate point with sleep 2s")
        robot.move_to_pose(
            [0, 0, dist],   
            [0, 0, 0],
            [1],
            velocities=velocities
        )
        # time.sleep(2)
        # base2gripper_transform, _, _ = robot.get_joint_transform('base_link', gripper_node)
        # print(f"transformed point3 : {base2gripper_transform * base_point}")
        diff = diff - dist
    
    # Now the gripper reached the grasping point and starts picking procedure
    robot.pickup(gripper_width)

    # Lifts the arm
    robot.move_to_position(lift_pos = robot.robot.get_six_joints()[1] + 0.2)
    # time.sleep(1)

    # Tucks the gripper so that while moving to place it wont collide with any obstacles
    robot.move_to_position(arm_pos = 0.01)
    # time.sleep(1)
    robot.move_to_position(wrist_pitch = 0.0)
    # time.sleep(1)
    robot.move_to_position(lift_pos = min(robot.robot.get_six_joints()[1], 0.9), wrist_yaw = 2.5)
    robot.move_to_position(lift_pos = min(robot.robot.get_six_joints()[1], 0.6))
    # time.sleep(1)

    # rotate the arm wrist onto the base
    # if abs(robot.robot.get_six_joints()[3] - 2.0) > 0.1:
    #     robot.move_to_position(wrist_yaw  = -2.0)
    #     # time.sleep(1)

    # # Put down the arm    
    # robot.move_to_position(lift_pos = 0.85)
    # if abs(robot.robot.get_six_joints()[3] - 2.0) < 0.1:
    #     robot.move_to_position(wrist_yaw = 2.5)
    # robot.move_to_position(lift_pos = 0.6)
    # time.sleep(1)
