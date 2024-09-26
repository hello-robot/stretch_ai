# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import time
from typing import Any, Dict

import numpy as np
import zmq

from stretch.agent import RobotClient
from stretch.core.parameters import Parameters
from stretch.dynav.communication_util import recv_array, send_array, send_everything
from stretch.dynav.ok_robot_hw.camera import RealSenseCamera
from stretch.dynav.ok_robot_hw.global_parameters import *
from stretch.dynav.ok_robot_hw.robot import HelloRobot as Manipulation_Wrapper
from stretch.dynav.ok_robot_hw.utils.grasper_utils import (
    capture_and_process_image,
    move_to_point,
    pickup,
)


class RobotAgentMDP:
    """Basic demo code. Collects everything that we need to make this work."""

    _retry_on_fail = False

    def __init__(
        self,
        robot: RobotClient,
        parameters: Dict[str, Any],
        ip: str,
        image_port: int = 5558,
        text_port: int = 5556,
        manip_port: int = 5557,
        re: int = 1,
        env_num: int = 1,
        test_num: int = 1,
        method: str = "dynamem",
    ):
        print("------------------------YOU ARE NOW RUNNING PEIQI CODES V3-----------------")
        self.re = re
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        if re == 1:
            stretch_gripper_max = 0.3
            end_link = "link_straight_gripper"
        else:
            stretch_gripper_max = 0.64
            end_link = "link_gripper_s3_body"
        self.transform_node = end_link
        self.manip_wrapper = Manipulation_Wrapper(
            self.robot, stretch_gripper_max=stretch_gripper_max, end_link=end_link
        )
        self.robot.move_to_nav_posture()

        self.normalize_embeddings = True
        self.pos_err_threshold = 0.35
        self.rot_err_threshold = 0.4
        self.obs_count = 0
        self.guarantee_instance_is_reachable = parameters.guarantee_instance_is_reachable

        self.image_sender = ImageSender(
            ip=ip, image_port=image_port, text_port=text_port, manip_port=manip_port
        )
        if method == "dynamem":
            from stretch.dynav.voxel_map_server import ImageProcessor as VoxelMapImageProcessor

            self.image_processor = VoxelMapImageProcessor(
                rerun=True, static=False, log="env" + str(env_num) + "_" + str(test_num)
            )  # type: ignore
        elif method == "mllm":
            from stretch.dynav.llm_server import ImageProcessor as mLLMImageProcessor

            self.image_processor = mLLMImageProcessor(
                rerun=True, static=False, log="env" + str(env_num) + "_" + str(test_num)
            )  # type: ignore

        self.look_around_times: list[float] = []
        self.execute_times: list[float] = []

        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    def look_around(self):
        print("*" * 10, "Look around to check", "*" * 10)
        for pan in [0.4, -0.4, -1.2]:
            for tilt in [-0.6]:
                self.robot.head_to(pan, tilt, blocking=True)
                self.update()

    def rotate_in_place(self):
        print("*" * 10, "Rotate in place", "*" * 10)
        xyt = self.robot.get_base_pose()
        self.robot.head_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for i in range(8):
            xyt[2] += 2 * np.pi / 8
            self.robot.navigate_to(xyt, blocking=True)
            self.update()

    def update(self):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = self.robot.get_observation()
        # self.image_sender.send_images(obs)
        self.obs_count += 1
        rgb, depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
        # start_time = time.time()
        self.image_processor.process_rgbd_images(rgb, depth, K, camera_pose)
        # end_time = time.time()
        # print('Image processing takes', end_time - start_time, 'seconds.')

    def execute_action(
        self,
        text: str,
    ):
        start_time = time.time()

        self.robot.look_front()
        self.look_around()
        self.robot.look_front()
        self.robot.switch_to_navigation_mode()

        start = self.robot.get_base_pose()
        # print("       Start:", start)
        # res = self.image_sender.query_text(text, start)
        res = self.image_processor.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            res = self.image_processor.process_text("", start)

        look_around_finish = time.time()
        look_around_take = look_around_finish - start_time
        print("Path planning takes ", look_around_take, " seconds.")
        # self.look_around_times.append(look_around_take)
        # print(self.look_around_times)
        # print(sum(self.look_around_times) / len(self.look_around_times))

        if len(res) > 0:
            print("Plan successful!")
            if len(res) >= 2 and np.isnan(res[-2]).all():
                # blocking = text != ''
                if len(res) > 2:
                    self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        blocking=True,
                    )

                execution_finish = time.time()
                execution_take = execution_finish - look_around_finish
                print("Executing action takes ", execution_take, " seconds.")
                self.execute_times.append(execution_take)
                print(self.execute_times)
                print(sum(self.execute_times) / len(self.execute_times))

                return True, res[-1]
            else:
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=False,
                )

                execution_finish = time.time()
                execution_take = execution_finish - look_around_finish
                print("Executing action takes ", execution_take, " seconds.")
                self.execute_times.append(execution_take)
                print(self.execute_times)
                print(sum(self.execute_times) / len(self.execute_times))

                return False, None
        else:
            print("Failed. Try again!")
            return None, None

    def run_exploration(self):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        status, _ = self.execute_action("")
        if status is None:
            print("Exploration failed! Perhaps nowhere to explore!")
            return False
        return True

    def navigate(self, text, max_step=10):
        finished = False
        step = 0
        end_point = None
        while not finished and step < max_step:
            print("*" * 20, step, "*" * 20)
            step += 1
            finished, end_point = self.execute_action(text)
            if finished is None:
                print("Navigation failed! The path might be blocked!")
                return None
        return end_point

    def place(self, text, init_tilt=INIT_HEAD_TILT, base_node=TOP_CAMERA_NODE):
        """
        An API for running placing. By calling this API, human will ask the robot to place whatever it holds
        onto objects specified by text queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - text: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        """
        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()
        self.manip_wrapper.move_to_position(head_pan=INIT_HEAD_PAN, head_tilt=init_tilt)
        camera = RealSenseCamera(self.robot)

        rotation, translation = capture_and_process_image(
            camera=camera,
            mode="place",
            obj=text,
            socket=self.image_sender.manip_socket,
            hello_robot=self.manip_wrapper,
        )

        if rotation is None:
            return False

        # lift arm to the top before the robot extends the arm, prepare the pre-placing gripper pose
        self.manip_wrapper.move_to_position(lift_pos=1.05)
        self.manip_wrapper.move_to_position(wrist_yaw=0, wrist_pitch=0)

        # Placing the object
        move_to_point(self.manip_wrapper, translation, base_node, self.transform_node, move_mode=0)
        self.manip_wrapper.move_to_position(gripper_pos=1, blocking=True)

        # Lift the arm a little bit, and rotate the wrist roll of the robot in case the object attached on the gripper
        self.manip_wrapper.move_to_position(
            lift_pos=min(self.manip_wrapper.robot.get_six_joints()[1] + 0.3, 1.1)
        )
        self.manip_wrapper.move_to_position(wrist_roll=2.5, blocking=True)
        self.manip_wrapper.move_to_position(wrist_roll=-2.5, blocking=True)

        # Wait for some time and shrink the arm back
        self.manip_wrapper.move_to_position(gripper_pos=1, lift_pos=1.05, arm_pos=0)
        self.manip_wrapper.move_to_position(wrist_pitch=-1.57)

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(
            base_trans=-self.manip_wrapper.robot.get_six_joints()[0]
        )
        return True

    def manipulate(self, text, init_tilt=INIT_HEAD_TILT, base_node=TOP_CAMERA_NODE):
        """
        An API for running manipulation. By calling this API, human will ask the robot to pick up objects
        specified by text queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - text: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        """

        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()

        gripper_pos = 1

        self.manip_wrapper.move_to_position(
            arm_pos=INIT_ARM_POS,
            head_pan=INIT_HEAD_PAN,
            head_tilt=init_tilt,
            gripper_pos=gripper_pos,
            lift_pos=INIT_LIFT_POS,
            wrist_pitch=INIT_WRIST_PITCH,
            wrist_roll=INIT_WRIST_ROLL,
            wrist_yaw=INIT_WRIST_YAW,
        )

        camera = RealSenseCamera(self.robot)

        rotation, translation, depth, width = capture_and_process_image(
            camera=camera,
            mode="pick",
            obj=text,
            socket=self.image_sender.manip_socket,
            hello_robot=self.manip_wrapper,
        )

        if rotation is None:
            return False

        if width < 0.05 and self.re == 3:
            gripper_width = 0.45
        elif width < 0.075 and self.re == 3:
            gripper_width = 0.6
        else:
            gripper_width = 1

        if input("Do you want to do this manipulation? Y or N ") != "N":
            pickup(
                self.manip_wrapper,
                rotation,
                translation,
                base_node,
                self.transform_node,
                gripper_depth=depth,
                gripper_width=gripper_width,
            )

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(
            base_trans=-self.manip_wrapper.robot.get_six_joints()[0]
        )

        return True

    def save(self):
        with self.image_processor.voxel_map_lock:
            self.image_processor.write_to_pickle()


class ImageSender:
    def __init__(
        self,
        stop_and_photo=False,
        ip="100.108.67.79",
        image_port=5560,
        text_port=5561,
        manip_port=5557,
        color_name="/camera/color",
        depth_name="/camera/aligned_depth_to_color",
        camera_name="/camera_pose",
        slop_time_seconds=0.05,
        queue_size=100,
    ):
        context = zmq.Context()
        self.img_socket = context.socket(zmq.REQ)
        self.img_socket.connect("tcp://" + str(ip) + ":" + str(image_port))
        self.text_socket = context.socket(zmq.REQ)
        self.text_socket.connect("tcp://" + str(ip) + ":" + str(text_port))
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + str(ip) + ":" + str(manip_port))

    def query_text(self, text, start):
        self.text_socket.send_string(text)
        self.text_socket.recv_string()
        send_array(self.text_socket, start)
        return recv_array(self.text_socket)

    def send_images(self, obs):
        rgb = obs.rgb
        depth = obs.depth
        camer_K = obs.camera_K
        camera_pose = obs.camera_pose
        # data = np.concatenate((depth.shape, rgb.flatten(), depth.flatten(), camer_K.flatten(), camera_pose.flatten()))
        # send_array(self.img_socket, data)
        send_everything(self.img_socket, rgb, depth, camer_K, camera_pose)
        # self.img_socket.recv_string()
