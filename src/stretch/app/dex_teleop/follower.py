#!/usr/bin/env python

import argparse
import pprint as pp
import threading
import time

import cv2
import numpy as np
import stretch_body.robot as rb
import zmq

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import stretch.app.dex_teleop.gripper_to_goal as gg
import stretch.app.dex_teleop.robot_move as rm
import stretch.motion.simple_ik as si
import stretch.utils.compression as compression
import stretch.utils.loop_stats as lt
from stretch.drivers.d405 import D405
from stretch.drivers.d435 import D435i
from stretch.utils.image import adjust_gamma, autoAdjustments_with_convertScaleAbs

HEAD_CONFIG = "head_config"
EE_POS = "wrist_position"


class DexTeleopFollower:
    def __init__(
        self,
        robot_speed: str = "slow",
        robot_allowed_to_move: bool = True,
        using_stretch_2: bool = False,
        manipulate_on_ground: bool = False,
        scaling: float = 0.5,
        gamma: float = 2.0,
        exposure: str = "low",
        send_port=5555,
        brighten_image: bool = False,
        use_remote_computer: bool = True,
    ):
        """
        Args:
          use_remote_computer(bool): is this process running on a different machine from the leader (default = True)
        """
        self.robot_speed = robot_speed
        self.robot_allowed_to_move = robot_allowed_to_move
        self.using_stretch_2 = using_stretch_2

        self.lift_middle = dt.get_lift_middle(manipulate_on_ground)
        self.center_configuration = dt.get_center_configuration(self.lift_middle)
        self.starting_configuration = dt.get_starting_configuration(self.lift_middle)

        self._robot_lock = threading.Lock()

        with self._robot_lock:
            ##########################################################
            # Prepare the robot last to avoid errors due to blocking calls
            # associated with other aspects of setting things up, such as
            # initializing SimpleIK.
            print("Connecting to Stretch body...")
            self.robot = None
            self.robot = rb.Robot()
            self.robot.startup()

            print("stretch_body file imported =", rb.__file__)
            transport_version = self.robot.arm.motor.transport.version
            print("stretch_body using transport version =", transport_version)

            self.robot_move = rm.RobotMove(self.robot, speed=robot_speed)
            self.robot_move.print_settings()

            self.robot_move.to_configuration(self.starting_configuration, speed="default")
            self.robot.push_command()
            self.robot.wait_command()

            # Set the current mobile base angle to be 0.0 radians.
            self.robot.base.reset_odometry()
            print("Stretch body is ready.")
            ##########################################################

        # Initialize IK
        self.simple_ik = si.SimpleIK()

        self.gripper_to_goal = gg.GripperToGoal(
            robot_speed=robot_speed,
            robot=self.robot,
            robot_move=self.robot_move,
            simple_ik=self.simple_ik,
            starting_configuration=self.starting_configuration,
            robot_allowed_to_move=robot_allowed_to_move,
            using_stretch_2=using_stretch_2,
        )

        # Define the center position for the wrist that corresponds with
        # the teleop origin.
        self.center_wrist_position = self.simple_ik.fk_rotary_base(self.center_configuration)

        # Create a socket for sending information
        self.context = zmq.Context()
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)

        if use_remote_computer:
            address = "tcp://*:" + str(send_port)
        else:
            address = "tcp://" + "127.0.0.1" + ":" + str(send_port)

        self.send_socket.bind(address)

        goal_recv_socket = self.context.socket(zmq.SUB)
        goal_recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        goal_recv_socket.setsockopt(zmq.SNDHWM, 1)
        goal_recv_socket.setsockopt(zmq.RCVHWM, 1)
        goal_recv_socket.setsockopt(zmq.CONFLATE, 1)
        # goal_recv_address = 'tcp://10.1.10.71:5555'
        goal_recv_address = "tcp://192.168.1.10:5555"
        goal_recv_socket.connect(goal_recv_address)

        # save the socket
        self.goal_recv_socket = goal_recv_socket
        self._done = False

        self.send_port = send_port
        self.exposure = exposure
        self.scaling = scaling
        self.gamma = gamma
        self.brighten_image = brighten_image
        self.ee_cam = D405(self.exposure)
        self.head_cam = D435i(exposure="auto")

        # Threads for sending and receiving commands
        self._send_thread = None
        self._recv_thread = None

    def spin_recv_commands(self):
        loop_timer = lt.LoopStats("dex_teleop_follower")
        print_timing = False
        print_goal = False

        while not self._done:
            loop_timer.mark_start()
            goal_dict = self.goal_recv_socket.recv_pyobj()
            if goal_dict:
                if print_goal:
                    print("goal_dict =")
                    pp.pprint(goal_dict)
                if HEAD_CONFIG in goal_dict:
                    self.set_head_config(goal_dict[HEAD_CONFIG])
                if EE_POS in goal_dict:
                    # We have received a spatial goal and will do IK to move the robot into position
                    self.gripper_to_goal.update_goal(**goal_dict)
            loop_timer.mark_end()
            if print_timing:
                loop_timer.pretty_print()

    def get_head_config(self):
        """Get the head configuration. Joints = head_pan, head_tilt."""
        with self._robot_lock:
            cfg = [
                self.robot.status["head"]["head_pan"],
                self.robot.status["head"]["head_tilt"],
            ]
        return cfg

    def set_head_config(self, config):
        """Sets the head configuration. Joints = head_pan, head_tilt."""
        assert len(config) == len(
            self.robot.head.joints
        ), f"must provide configs for each of {self.robot.head.joints}"
        with self._robot_lock:
            for joint, pos in zip(self.robot.head.joints, config):
                self.robot.head.move_to(joint, pos)
            self.robot.push_command()

    def robot_head_forward(self):
        self.set_head_config([0, 0])

    def _get_images(self, from_head: bool = False, verbose: bool = False):
        """Get the images from the end effector camera"""
        if from_head:
            if verbose:
                print("Getting head images:")
            depth_frame, color_frame = self.head_cam.get_frames()
        else:
            if verbose:
                print("Getting end effector images:")
            depth_frame, color_frame = self.ee_cam.get_frames()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if verbose:
            print(f"{depth_image.shape=} {color_image.shape=}")

        if self.gamma != 1.0:
            color_image = adjust_gamma(color_image, self.gamma)
            if verbose:
                print(f" - gamma adjustment {self.gamma}")

        if self.scaling != 1.0:
            color_image = cv2.resize(
                color_image,
                (0, 0),
                fx=self.scaling,
                fy=self.scaling,
                interpolation=cv2.INTER_AREA,
            )
            depth_image = cv2.resize(
                depth_image,
                (0, 0),
                fx=self.scaling,
                fy=self.scaling,
                interpolation=cv2.INTER_NEAREST,
            )
            if verbose:
                print(f" - scaled by {self.scaling}")

        if verbose:
            print(f"{depth_image.shape=} {color_image.shape=}")
        return depth_image, color_image

    def spin_send_images(self, verbose: bool = False):
        """Send the images here as well"""
        loop_timer = lt.LoopStats("d405_sender", target_loop_rate=15)
        depth_camera_info, color_camera_info = self.ee_cam.get_camera_infos()
        head_depth_camera_info, head_color_camera_info = self.head_cam.get_camera_infos()
        depth_scale = self.ee_cam.get_depth_scale()
        head_depth_scale = self.head_cam.get_depth_scale()

        while not self._done:
            loop_timer.mark_start()
            depth_image, color_image = self._get_images(from_head=False, verbose=verbose)
            head_depth_image, head_color_image = self._get_images(from_head=True, verbose=verbose)

            # import timeit
            # t0 = timeit.default_timer()
            # data1 = compression.zip(color_image)
            # t1 = timeit.default_timer()
            # data2 = compression.to_webp(color_image)
            # t2 = timeit.default_timer()
            # print(t1 - t0, len(data1), t2 -t1, len(data2))

            if self.brighten_image:
                color_image = autoAdjustments_with_convertScaleAbs(color_image)

            config = self.gripper_to_goal.get_current_config()
            ee_pos, ee_rot = self.gripper_to_goal.get_current_ee_pose()

            d405_output = {
                "ee_cam/color_camera_info": color_camera_info,
                "ee_cam/depth_camera_info": depth_camera_info,
                "ee_cam/color_image": compression.to_webp(color_image),
                "ee_cam/depth_image": compression.to_webp(depth_image),
                "ee_cam/color_image": color_image,
                "ee_cam/depth_image": depth_image,
                "ee_cam/depth_scale": depth_scale,
                "ee_cam/image_gamma": self.gamma,
                "ee_cam/image_scaling": self.scaling,
                "head_cam/color_camera_info": head_color_camera_info,
                "head_cam/depth_camera_info": head_depth_camera_info,
                # "head_cam/color_image": compression.zip(head_color_image),
                # "head_cam/depth_image": compression.zip(head_depth_image),
                "head_cam/depth_scale": head_depth_scale,
                "head_cam/image_gamma": self.gamma,
                "head_cam/image_scaling": self.scaling,
                "robot/config": config,
                "robot/ee_position": ee_pos,
                "robot/ee_rotation": ee_rot,
            }

            self.send_socket.send_pyobj(d405_output)

            loop_timer.mark_end()
            if True or verbose:
                loop_timer.pretty_print()

    def start(self):
        """Start threads for sending and receiving commands."""
        self._send_thread = threading.Thread(target=self.spin_send_images)
        self._recv_thread = threading.Thread(target=self.spin_recv_commands)
        self._done = False
        self._send_thread.start()
        self._recv_thread.start()

    def __del__(self):
        self._done = True
        if self._send_thread:
            self._send_thread.terminate()
            self._recv_thread.terminate()
            self._send_thread.join()
            self._recv_thread.join()
        self.goal_recv_socket.close()
        self.send_socket.close()
        self.context.term()

    def spin(self):
        """Main thread to start both processes and wait"""
        self.start()
        print("Running threads to publish images and receive motor commands!")
        while True:
            time.sleep(0.1)


def main(args):
    use_fastest_mode = args.fast
    manipulate_on_ground = args.ground
    left_handed = args.left
    using_stretch_2 = args.stretch_2
    slide_lift_range = args.slide_lift_range

    # The 'default', 'slow', 'fast', and 'max' options are defined by
    # Hello Robot. The 'fastest_stretch_2' option has been specially tuned for
    # this application.
    #
    # WARNING: 'fastest_stretch_*' have velocities and accelerations that exceed
    # the factory 'max' values defined by Hello Robot.
    if use_fastest_mode:
        if using_stretch_2:
            robot_speed = "fastest_stretch_2"
        else:
            robot_speed = "fastest_stretch_3"
    else:
        robot_speed = "slow"

    # Note on control here
    print("Running with robot_speed =", robot_speed)

    follower = DexTeleopFollower(
        robot_speed,
        manipulate_on_ground=manipulate_on_ground,
        robot_allowed_to_move=True,
        using_stretch_2=using_stretch_2,
        scaling=args.scaling,
        gamma=args.gamma,
        exposure=args.exposure,
        send_port=args.send_port,
    )
    follower.spin()


if __name__ == "__main__":

    args = dt.get_arg_parser().parse_args()
    main(args)
