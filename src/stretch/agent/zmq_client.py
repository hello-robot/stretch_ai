# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 chris paxton under MIT license

import threading
import time
import timeit
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import zmq
from termcolor import colored

import stretch.motion.constants as constants
import stretch.motion.conversions as conversions
import stretch.utils.compression as compression
from stretch.core.interfaces import ContinuousNavigationAction, Observations
from stretch.core.parameters import Parameters, get_parameters
from stretch.core.robot import AbstractRobotClient
from stretch.motion import PlanResult
from stretch.motion.kinematics import HelloStretchIdx, HelloStretchKinematics
from stretch.utils.geometry import angle_difference, posquat2sophus, sophus2posquat
from stretch.utils.image import Camera
from stretch.utils.logger import Logger
from stretch.utils.memory import lookup_address
from stretch.utils.point_cloud import show_point_cloud

logger = Logger(__name__)

# TODO: debug code - remove later if necessary
# import faulthandler
# faulthandler.enable()


class HomeRobotZmqClient(AbstractRobotClient):

    update_base_pose_from_full_obs: bool = False
    num_state_report_steps: int = 10000

    _head_pan_min = -np.pi
    _head_pan_max = np.pi / 4
    _head_tilt_min = -np.pi
    _head_tilt_max = 0

    def _create_recv_socket(
        self,
        port: int,
        robot_ip: str,
        use_remote_computer: bool,
        message_type: Optional[str] = "observations",
    ):
        # Receive state information
        recv_socket = self.context.socket(zmq.SUB)
        recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        recv_socket.setsockopt(zmq.SNDHWM, 1)
        recv_socket.setsockopt(zmq.RCVHWM, 1)
        recv_socket.setsockopt(zmq.CONFLATE, 1)

        recv_address = lookup_address(robot_ip, use_remote_computer) + ":" + str(port)
        print(f"Connecting to {recv_address} to receive {message_type}...")
        recv_socket.connect(recv_address)

        return recv_socket

    def get_zmq_context(self) -> zmq.Context:
        """Get the ZMQ context for the client.

        Returns:
            zmq.Context: The ZMQ context
        """
        return self.context

    def __init__(
        self,
        robot_ip: str = "",
        recv_port: int = 4401,
        send_port: int = 4402,
        recv_state_port: int = 4403,
        recv_servo_port: int = 4404,
        pub_obs_port: int = 4450,
        parameters: Parameters = None,
        use_remote_computer: bool = True,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
        start_immediately: bool = True,
        enable_rerun_server: bool = True,
        resend_all_actions: bool = False,
        publish_observations: bool = False,
    ):
        """
        Create a client to communicate with the robot over ZMQ.

        Args:
            robot_ip: The IP address of the robot
            recv_port: The port to receive observations on
            send_port: The port to send actions to on the robot
            use_remote_computer: Whether to use a remote computer to connect to the robot
            urdf_path: The path to the URDF file for the robot
            ik_type: The type of IK solver to use
            visualize_ik: Whether to visualize the IK solution
            grasp_frame: The frame to use for grasping
            ee_link_name: The name of the end effector link
            manip_mode_controlled_joints: The joints to control in manipulation mode
        """
        self.recv_port = recv_port
        self.send_port = send_port
        self.reset()

        # Load parameters
        if parameters is None:
            parameters = get_parameters("default_planner.yaml")
        self._parameters = parameters

        # Variables we set here should not change
        self._iter = -1  # Tracks number of actions set, never reset this
        self._seq_id = 0  # Number of messages we received
        self._started = False

        # Resend all actions immediately - helps if we are losing packets or something?
        self._resend_all_actions = resend_all_actions
        self._publish_observations = (
            publish_observations or self.parameters["agent"]["use_realtime_updates"]
        )

        self._moving_threshold = parameters["motion"]["moving_threshold"]
        self._angle_threshold = parameters["motion"]["angle_threshold"]
        self._min_steps_not_moving = parameters["motion"]["min_steps_not_moving"]

        # Read in joint tolerances from config file
        self._head_pan_tolerance = float(parameters["motion"]["joint_tolerance"]["head_pan"])
        self._head_tilt_tolerance = float(parameters["motion"]["joint_tolerance"]["head_tilt"])
        self._head_not_moving_tolerance = float(
            parameters["motion"]["joint_thresholds"]["head_not_moving_tolerance"]
        )
        self._arm_joint_tolerance = float(parameters["motion"]["joint_tolerance"]["arm"])
        self._lift_joint_tolerance = float(parameters["motion"]["joint_tolerance"]["lift"])
        self._base_x_joint_tolerance = float(parameters["motion"]["joint_tolerance"]["base_x"])
        self._wrist_roll_joint_tolerance = float(
            parameters["motion"]["joint_tolerance"]["wrist_roll"]
        )
        self._wrist_pitch_joint_tolerance = float(
            parameters["motion"]["joint_tolerance"]["wrist_pitch"]
        )
        self._wrist_yaw_joint_tolerance = float(
            parameters["motion"]["joint_tolerance"]["wrist_yaw"]
        )

        # Robot model
        self._robot_model = HelloStretchKinematics(
            urdf_path=urdf_path,
            ik_type=ik_type,
            visualize=visualize_ik,
            grasp_frame=grasp_frame,
            ee_link_name=ee_link_name,
            manip_mode_controlled_joints=manip_mode_controlled_joints,
        )

        # Create ZMQ sockets
        self.context = zmq.Context()

        print("-------- HOME-ROBOT ROS2 ZMQ CLIENT --------")
        self.recv_socket = self._create_recv_socket(
            self.recv_port, robot_ip, use_remote_computer, message_type="observations"
        )
        self.recv_state_socket = self._create_recv_socket(
            recv_state_port, robot_ip, use_remote_computer, message_type="low level state"
        )
        self.recv_servo_socket = self._create_recv_socket(
            recv_servo_port, robot_ip, use_remote_computer, message_type="visual servoing data"
        )

        # Send actions back to the robot for execution
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)

        self.send_address = (
            lookup_address(robot_ip, use_remote_computer) + ":" + str(self.send_port)
        )

        print(f"Connecting to {self.send_address} to send action messages...")
        self.send_socket.connect(self.send_address)
        print("...connected.")

        self._obs_lock = Lock()
        self._act_lock = Lock()
        self._state_lock = Lock()
        self._servo_lock = Lock()

        if enable_rerun_server:
            from stretch.visualization.rerun import RerunVisualizer

            self._rerun = RerunVisualizer()
        else:
            self._rerun = None
            self._rerun_thread = None

        if start_immediately:
            self.start()

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    def get_ee_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the RGB and depth images from the end effector camera.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The RGB and depth images
        """
        with self._servo_lock:
            if self._servo is None:
                return None, None
            rgb = self._servo_obs["ee_rgb"]
            depth = self._servo_obs["ee_depth"]
        return rgb, depth

    def get_head_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the RGB and depth images from the head camera.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The RGB and depth images
        """
        with self._servo_lock:
            if self._servo is None:
                return None, None
            rgb = self._servo["head_rgb"]
            depth = self._servo["head_depth"]
        return rgb, depth

    def get_joint_state(self, timeout: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the current joint positions, velocities, and efforts"""
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None, None, None
            joint_positions = self._state["joint_positions"]
            joint_velocities = self._state["joint_velocities"]
            joint_efforts = self._state["joint_efforts"]
        return joint_positions, joint_velocities, joint_efforts

    def get_joint_positions(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current joint positions"""
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_positions = self._state["joint_positions"]
        return joint_positions

    def get_six_joints(self, timeout: float = 5.0) -> np.ndarray:
        """Get the six major joint positions"""
        joint_positions = self.get_joint_positions(timeout=timeout)
        return np.array(self._extract_joint_pos(joint_positions))

    def get_joint_velocities(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current joint velocities"""
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_velocities = self._state["joint_velocities"]
        return joint_velocities

    def get_joint_efforts(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current joint efforts from the robot.

        Args:
            timeout: How long to wait for the observation

        Returns:
            np.ndarray: The joint efforts as an array of floats
        """

        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_efforts = self._state["joint_efforts"]
        return joint_efforts

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current pose of the base.

        Args:
            timeout: How long to wait for the observation

        Returns:
            np.ndarray: The base pose as [x, y, theta]
        """
        t0 = timeit.default_timer()
        if self.update_base_pose_from_full_obs:
            with self._obs_lock:
                while self._obs is None:
                    time.sleep(0.01)
                    if timeit.default_timer() - t0 > timeout:
                        logger.error("Timeout waiting for observation")
                        return None
                gps = self._obs["gps"]
                compass = self._obs["compass"]
                xyt = np.concatenate([gps, compass], axis=-1)
        else:
            with self._state_lock:
                while self._state is None:
                    time.sleep(1e-4)
                    if timeit.default_timer() - t0 > timeout:
                        logger.error("Timeout waiting for state message")
                        return None
                xyt = self._state["base_pose"]
        return xyt

    def get_pan_tilt(self):
        """Get the current pan and tilt of the head.

        Returns:
            Tuple[float, float]: The pan and tilt angles
        """

        joint_positions, _, _ = self.get_joint_state()
        return joint_positions[HelloStretchIdx.HEAD_PAN], joint_positions[HelloStretchIdx.HEAD_TILT]

    def get_gripper_position(self):
        """Get the current position of the gripper.

        Returns:
            float: The position of the gripper
        """
        joint_state = self.get_joint_positions()
        return joint_state[HelloStretchIdx.GRIPPER]

    def get_ee_pose(self, matrix=False, link_name=None, q=None):
        """Get the current pose of the end effector.

        Args:
            matrix: Whether to return the pose as a matrix
            link_name: The name of the link to get the pose of
            q: The joint positions to use

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The position and orientation of the end effector
        """
        if q is None:
            q = self.get_joint_positions()
        pos, quat = self._robot_model.manip_fk(q, node=link_name)

        if matrix:
            pose = posquat2sophus(pos, quat)
            return pose.matrix()
        else:
            return pos, quat

    def get_frame_pose(self, q: Union[np.ndarray, dict], node_a: str, node_b: str) -> np.ndarray:
        """Get the pose of frame b relative to frame a.

        Args:
            q: The joint positions
            node_a: The name of the first frame
            node_b: The name of the second frame

        Returns:
            np.ndarray: The pose of frame b relative to frame a as a 4x4 matrix
        """
        # TODO: get this working properly and update the documentation
        return self._robot_model.manip_ik_solver.get_frame_pose(q, node_a, node_b)

    def solve_ik(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        initial_cfg: np.ndarray = None,
        debug: bool = False,
        custom_ee_frame: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics appropriately (or at least try to) and get the joint position
        that we will be moving to.

        Note: When relative==True, the delta orientation is still defined in the world frame

        Returns None if no solution is found, else returns an executable solution
        """

        pos_ee_curr, quat_ee_curr = self.get_ee_pose()
        if quat is None:
            quat = quat_ee_curr

        # Compute IK goal: pose relative to base
        pose_desired = posquat2sophus(np.array(pos), np.array(quat))

        pose_base2ee_desired = pose_desired

        pos_ik_goal, quat_ik_goal = sophus2posquat(pose_base2ee_desired)

        # Execute joint command
        if debug:
            print("=== EE goto command ===")
            print(f"Initial EE pose: pos={pos_ee_curr}; quat={quat_ee_curr}")
            print(f"Input EE pose: pos={np.array(pos)}; quat={np.array(quat)}")
            print(f"Desired EE pose: pos={pos_ik_goal}; quat={quat_ik_goal}")

        # Perform IK
        full_body_cfg, ik_success, ik_debug_info = self._robot_model.manip_ik(
            (pos_ik_goal, quat_ik_goal),
            q0=initial_cfg,
            custom_ee_frame=custom_ee_frame,
        )

        # Expected to return None if we did not get a solution
        if not ik_success or full_body_cfg is None:
            return None
        # Return a valid solution to the IK problem here
        return full_body_cfg

    def _extract_joint_pos(self, q):
        """Helper to convert from the general-purpose config including full robot state, into the command space used in just the manip controller. Extracts just lift/arm/wrist information."""
        return [
            q[HelloStretchIdx.BASE_X],
            q[HelloStretchIdx.LIFT],
            q[HelloStretchIdx.ARM],
            q[HelloStretchIdx.WRIST_YAW],
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_ROLL],
        ]

    def get_pose_graph(self) -> np.ndarray:
        """Get the robot's SLAM pose graph"""
        return self._pose_graph

    def robot_to(self, joint_angles: np.ndarray, blocking: bool = False, timeout: float = 10.0):
        """Move the robot to a particular joint configuration."""
        next_action = {"joint": joint_angles, "manip_blocking": blocking}
        self.send_action(next_action=next_action, timeout=timeout)

    def head_to(
        self, head_pan: float, head_tilt: float, blocking: bool = False, timeout: float = 10.0
    ):
        """Move the head to a particular configuration."""
        if head_pan < self._head_pan_min or head_pan > self._head_pan_max:
            logger.warning(
                "Head pan is restricted to be between {self._head_pan_min} and {self._head_pan_max} for safety: was {head_pan}"
            )
        if head_tilt > self._head_tilt_max or head_tilt < self._head_tilt_min:
            logger.warning(
                f"Head tilt is restricted to be between {self._head_tilt_min} and {self._head_tilt_max} for safety: was{head_tilt}"
            )
        head_pan = np.clip(head_pan, self._head_pan_min, self._head_pan_max)
        head_tilt = np.clip(head_tilt, -np.pi / 2, 0)
        next_action = {"head_to": [float(head_pan), float(head_tilt)], "manip_blocking": blocking}
        sent = self.send_action(next_action, timeout=timeout)

        if blocking:
            step = sent["step"]
            whole_body_q = np.zeros(self._robot_model.dof, dtype=np.float32)
            whole_body_q[HelloStretchIdx.HEAD_PAN] = float(head_pan)
            whole_body_q[HelloStretchIdx.HEAD_TILT] = float(head_tilt)
            time.sleep(0.25)
            self._wait_for_head(whole_body_q, block_id=step)
            time.sleep(0.25)

    def look_front(self, blocking: bool = True, timeout: float = 10.0):
        """Let robot look to its front."""
        self.head_to(
            constants.look_front[0], constants.look_front[1], blocking=blocking, timeout=timeout
        )

    def look_at_ee(self, blocking: bool = True, timeout: float = 10.0):
        """Let robot look to its arm."""
        self.head_to(
            constants.look_at_ee[0], constants.look_at_ee[1], blocking=blocking, timeout=timeout
        )

    def arm_to(
        self,
        joint_angles: Optional[np.ndarray] = None,
        gripper: float = None,
        head: Optional[np.ndarray] = None,
        blocking: bool = False,
        timeout: float = 10.0,
        verbose: bool = False,
        min_time: float = 2.5,
        **config,
    ) -> bool:
        """Move the arm to a particular joint configuration.

        Args:
            joint_angles: 6 or Nx6 array of the joint angles to move to
            blocking: Whether to block until the motion is complete
            timeout: How long to wait for the motion to complete
            verbose: Whether to print out debug information
            **config: arm configuration options; maps joints to values.

        Returns:
            bool: Whether the motion was successful
        """
        if not self.in_manipulation_mode():
            raise ValueError("Robot must be in manipulation mode to move the arm")
        if isinstance(joint_angles, list):
            joint_angles = np.array(joint_angles)
        if joint_angles is None:
            assert (
                config is not None and len(config.keys()) > 0
            ), "Must provide joint angles array or specific joint values as params"
            joint_positions = self.get_joint_positions()
            joint_angles = conversions.config_to_manip_command(joint_positions)
        elif len(joint_angles) > 6:
            print(
                "[WARNING] arm_to: attempting to convert from full robot state to 6dof manipulation state."
            )
            joint_angles = conversions.config_to_manip_command(joint_angles)
        if head is not None:
            assert len(head) == 2, "Head must be a 2D vector of pan and tilt"

        elif len(joint_angles) < 6:
            raise ValueError(
                "joint_angles must be 6 dimensional: base_x, lift, arm, wrist roll, wrist pitch, wrist yaw"
            )
        if config is not None and len(config.keys()) > 0:
            # Convert joint names to indices and update joint angles
            for joint, value in config.items():
                joint_angles[conversions.get_manip_joint_idx(joint)] = value
        # Make sure it's all the right size
        assert (
            len(joint_angles) == 6
        ), "joint angles must be 6 dimensional: base_x, lift, arm, wrist roll, wrist pitch, wrist yaw"

        # Create and send the action dictionary
        _next_action = {"joint": joint_angles}
        if gripper is not None:
            _next_action["gripper"] = gripper
        if head is not None:
            _next_action["head_to"] = head
        else:
            # TODO: remove this once we no longer need to specify all joints for arm_to
            # If head is not specified, we need to set it to the right head position
            # In this case, we assume if moving arm you should look at ee
            _next_action["head_to"] = constants.look_at_ee
            # cur_pan, cur_tilt = self.get_pan_tilt()
            # _next_action["head_to"] = np.array([cur_pan, cur_tilt])
        _next_action["manip_blocking"] = blocking
        self.send_action(_next_action)

        # Handle blocking
        steps = 0
        if blocking:
            t0 = timeit.default_timer()
            while not self._finish:

                if steps % 10 == 9:
                    # Resend the action until we get there
                    self.send_action(_next_action)
                    if verbose:
                        print("Resending action", joint_angles)

                joint_state, joint_velocities, _ = self.get_joint_state()
                if joint_state is None:
                    time.sleep(0.01)
                    continue

                arm_diff = np.abs(joint_state[HelloStretchIdx.ARM] - joint_angles[2])
                lift_diff = np.abs(joint_state[HelloStretchIdx.LIFT] - joint_angles[1])
                base_x_diff = np.abs(joint_state[HelloStretchIdx.BASE_X] - joint_angles[0])
                wrist_roll_diff = np.abs(
                    angle_difference(joint_state[HelloStretchIdx.WRIST_ROLL], joint_angles[3])
                )
                wrist_pitch_diff = np.abs(
                    angle_difference(joint_state[HelloStretchIdx.WRIST_PITCH], joint_angles[4])
                )
                wrist_yaw_diff = np.abs(
                    angle_difference(joint_state[HelloStretchIdx.WRIST_YAW], joint_angles[5])
                )
                if verbose:
                    print(
                        f"{arm_diff=}, {lift_diff=}, {base_x_diff=}, {wrist_roll_diff=}, {wrist_pitch_diff=}, {wrist_yaw_diff=}"
                    )

                t1 = timeit.default_timer()
                if (
                    (arm_diff < self._arm_joint_tolerance)
                    and (lift_diff < self._lift_joint_tolerance)
                    and (base_x_diff < self._base_x_joint_tolerance)
                    and (wrist_roll_diff < self._wrist_roll_joint_tolerance)
                    and (wrist_pitch_diff < self._wrist_pitch_joint_tolerance)
                    and (wrist_yaw_diff < self._wrist_yaw_joint_tolerance)
                ):
                    return True
                elif t1 - t0 > min_time and np.linalg.norm(joint_velocities) < 0.01:
                    logger.info("Arm not moving, we are done")
                    logger.info("Arm joint velocities", joint_velocities)
                    logger.info(t1 - t0)
                    # Arm stopped moving but did not reach goal
                    return False
                else:
                    if verbose:
                        print(
                            f"{arm_diff=}, {lift_diff=}, {base_x_diff=}, {wrist_roll_diff=}, {wrist_pitch_diff=}, {wrist_yaw_diff=}"
                        )
                time.sleep(0.01)

                if t1 - t0 > timeout:
                    logger.error("Timeout waiting for arm to move")
                    break
                steps += 1
            return False
        return True

    def navigate_to(
        self,
        xyt: Union[np.ndarray, ContinuousNavigationAction],
        relative=False,
        blocking=False,
        timeout: float = 10.0,
        verbose: bool = False,
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        if isinstance(xyt, ContinuousNavigationAction):
            _xyt = xyt.xyt
        else:
            _xyt = xyt
        assert len(_xyt) == 3, "xyt must be a vector of size 3"
        next_action = {"xyt": _xyt, "nav_relative": relative, "nav_blocking": blocking}
        if self._rerun:
            self._rerun.update_nav_goal(_xyt)
        # If we are not in navigation mode, switch to it
        # Send an action to the robot
        # Resend it to make sure it arrives, if we are not making a relative motion
        # If we are blocking, wait for the action to complete with a timeout
        self.send_action(next_action, timeout=timeout, verbose=verbose, force_resend=(not relative))

    def set_velocity(self, v: float, w: float):
        """Move to xyt in global coordinates or relative coordinates."""
        next_action = {"v": v, "w": w}
        self.send_action(next_action)

    def reset(self):
        """Reset everything in the robot's internal state"""
        self._control_mode = None
        self._obs = None  # Full observation includes high res images and camera pose, no EE camera
        self._pose_graph = None
        self._state = None  # Low level state includes joint angles and base XYT
        self._servo = None  # Visual servoing state includes smaller images
        self._thread = None
        self._finish = False
        self._last_step = -1

    def open_gripper(
        self, blocking: bool = True, timeout: float = 10.0, verbose: bool = False
    ) -> bool:
        """Open the gripper based on hard-coded presets."""
        gripper_target = self._robot_model.GRIPPER_OPEN
        print("[ZMQ CLIENT] Opening gripper to", gripper_target)
        self.gripper_to(gripper_target, blocking=False)
        if blocking:
            t0 = timeit.default_timer()
            while not self._finish:
                self.gripper_to(gripper_target, blocking=False)
                joint_state = self.get_joint_positions()
                if joint_state is None:
                    continue
                if verbose:
                    print("Opening gripper:", joint_state[HelloStretchIdx.GRIPPER])
                gripper_err = np.abs(joint_state[HelloStretchIdx.GRIPPER] - gripper_target)
                if gripper_err < 0.1:
                    return True
                t1 = timeit.default_timer()
                if t1 - t0 > timeout:
                    print("[ZMQ CLIENT] Timeout waiting for gripper to open")
                    break
                self.gripper_to(gripper_target, blocking=False)
                time.sleep(0.01)
            return False
        return True

    def close_gripper(
        self,
        loose: bool = False,
        blocking: bool = True,
        timeout: float = 10.0,
        verbose: bool = False,
    ) -> bool:
        """Close the gripper based on hard-coded presets."""
        gripper_target = (
            self._robot_model.GRIPPER_CLOSED_LOOSE if loose else self._robot_model.GRIPPER_CLOSED
        )
        print("[ZMQ CLIENT] Closing gripper to", gripper_target)
        self.gripper_to(gripper_target, blocking=False)
        if blocking:
            t0 = timeit.default_timer()
            while not self._finish:
                joint_state = self.get_joint_positions()
                if joint_state is None:
                    continue
                gripper_err = np.abs(joint_state[HelloStretchIdx.GRIPPER] - gripper_target)
                if verbose:
                    print("Closing gripper:", gripper_err, gripper_target)
                if gripper_err < 0.1:
                    return True
                t1 = timeit.default_timer()
                if t1 - t0 > timeout:
                    print("[ZMQ CLIENT] Timeout waiting for gripper to close")
                    break
                self.gripper_to(gripper_target, blocking=False)
                time.sleep(0.01)
            return False
        return True

    def gripper_to(self, target: float, blocking: bool = True):
        """Send the gripper to a target position."""
        next_action = {"gripper": target, "gripper_blocking": blocking}
        self.send_action(next_action)
        if blocking:
            time.sleep(2.0)

    def switch_to_navigation_mode(self):
        """Velocity control of the robot base."""
        next_action = {"control_mode": "navigation"}
        action = self.send_action(next_action)
        self._wait_for_mode("navigation", resend_action=action)
        assert self.in_navigation_mode()

    def switch_to_manipulation_mode(self, verbose: bool = False):
        """Move the robot to manipulation mode.

        Args:
            verbose: Whether to print out debug information
        """
        next_action = {"control_mode": "manipulation"}
        action = self.send_action(next_action)
        time.sleep(0.1)
        if verbose:
            logger.info("Waiting for manipulation mode")
        self._wait_for_mode("manipulation", resend_action=action, verbose=verbose)
        assert self.in_manipulation_mode()

    def move_to_nav_posture(self):
        """Move the robot to the navigation posture. This is where the head is looking forward and the arm is tucked in."""
        next_action = {"posture": "navigation", "step": self._iter}
        self.send_action(next_action)
        self._wait_for_head(constants.STRETCH_NAVIGATION_Q, resend_action={"posture": "navigation"})
        self._wait_for_mode("navigation")
        # self._wait_for_mode("navigation", resend_action=next_action)
        # self._wait_for_arm(constants.STRETCH_NAVIGATION_Q)
        assert self.in_navigation_mode()

    def move_to_manip_posture(self):
        """This is the pregrasp posture where the head is looking down and right and the arm is tucked in."""
        next_action = {"posture": "manipulation", "step": self._iter}
        self.send_action(next_action)
        time.sleep(0.1)
        self._wait_for_head(constants.STRETCH_PREGRASP_Q, resend_action={"posture": "manipulation"})
        self._wait_for_mode("manipulation")
        # self._wait_for_arm(constants.STRETCH_PREGRASP_Q)
        assert self.in_manipulation_mode()

    def _wait_for_head(
        self,
        q: np.ndarray,
        timeout: float = 3.0,
        min_wait_time: float = 0.5,
        resend_action: Optional[dict] = None,
        block_id: int = -1,
        verbose: bool = False,
    ) -> None:
        """Wait for the head to move to a particular configuration."""
        t0 = timeit.default_timer()
        at_goal = False

        # Wait for the head to move
        # If the head is not moving, we are done
        # Head must be stationary for at least min_wait_time
        prev_joint_positions = None
        prev_t = None
        while not self._finish:
            joint_positions, joint_velocities, _ = self.get_joint_state()

            if joint_positions is None:
                continue

            # if self._last_step < block_id:
            #     # TODO: remove debug info
            #     print("Waiting for step", block_id, "to be processed; currently on:", self._last_step)
            #     time.sleep(0.05)
            #     continue

            pan_err = np.abs(
                joint_positions[HelloStretchIdx.HEAD_PAN] - q[HelloStretchIdx.HEAD_PAN]
            )
            tilt_err = np.abs(
                joint_positions[HelloStretchIdx.HEAD_TILT] - q[HelloStretchIdx.HEAD_TILT]
            )
            head_speed = np.linalg.norm(
                joint_velocities[HelloStretchIdx.HEAD_PAN : HelloStretchIdx.HEAD_TILT]
            )

            if prev_joint_positions is not None:
                head_speed_v2 = np.linalg.norm(
                    joint_positions[HelloStretchIdx.HEAD_PAN : HelloStretchIdx.HEAD_TILT]
                    - prev_joint_positions[HelloStretchIdx.HEAD_PAN : HelloStretchIdx.HEAD_TILT]
                ) / (timeit.default_timer() - prev_t)
            else:
                head_speed_v2 = float("inf")

            # Take the max of the two speeds
            # This is to handle the case where we're getting weird measurements
            head_speed = max(head_speed, head_speed_v2)

            # Save the current joint positions to compute speed
            prev_joint_positions = joint_positions
            prev_t = timeit.default_timer()

            if verbose:
                print("Waiting for head to move", pan_err, tilt_err, "head speed =", head_speed)
            if head_speed > self._head_not_moving_tolerance:
                at_goal = False
            elif pan_err < self._head_pan_tolerance and tilt_err < self._head_tilt_tolerance:
                at_goal = True
                at_goal_t = timeit.default_timer()
            elif resend_action is not None:
                self.send_socket.send_pyobj(resend_action)
            else:
                at_goal = False

            if (
                at_goal
                and timeit.default_timer() - at_goal_t > min_wait_time
                and head_speed < self._head_not_moving_tolerance
            ):
                break

            t1 = timeit.default_timer()
            if t1 - t0 > min_wait_time and head_speed < self._head_not_moving_tolerance:
                if verbose:
                    print("Head not moving, we are done")
                break

            if t1 - t0 > timeout:
                print("Timeout waiting for head to move")
                break
            time.sleep(0.01)

    def _wait_for_arm(
        self, q: np.ndarray, timeout: float = 10.0, resend_action: Optional[dict] = None
    ) -> bool:
        """Wait for the arm to move to a particular configuration.

        Args:
            q(np.ndarray): The target joint angles
            timeout(float): How long to wait for the arm to move
            resend_action(dict): The action to resend if the arm is not moving. If none, do not resend.

        Returns:
            bool: Whether the arm successfully moved to the target configuration
        """
        t0 = timeit.default_timer()
        while not self._finish:
            joint_positions, joint_velocities, _ = self.get_joint_state()
            if joint_positions is None:
                continue

            arm_diff = np.abs(joint_positions[HelloStretchIdx.ARM] - q[HelloStretchIdx.ARM])
            lift_diff = np.abs(joint_positions[HelloStretchIdx.LIFT] - q[HelloStretchIdx.LIFT])

            if arm_diff < self._arm_joint_tolerance and lift_diff < self._lift_joint_tolerance:
                return True

            if resend_action is not None:
                self.send_socket.send_pyobj(resend_action)

            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                logger.error(
                    f"Timeout waiting for arm to move to arm={q[HelloStretchIdx.ARM]}, lift={q[HelloStretchIdx.LIFT]}"
                )
                return False

        # This should never happen
        return False

    def _wait_for_mode(
        self,
        mode,
        resend_action: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        timeout: float = 20.0,
        time_required: float = 0.05,
    ) -> bool:
        """
        Wait for the robot to switch to a particular control mode. Will throw an exception if mode switch fails; probably means a packet was dropped.

        Args:
            mode(str): The mode to wait for
            resend_action(dict): The action to resend if the robot is not moving. If none, do not resend.
            verbose(bool): Whether to print out debug information
            timeout(float): How long to wait for the robot to switch modes

        Returns:
            bool: Whether the robot successfully switched to the target mode
        """
        t0 = timeit.default_timer()
        mode_t0 = None
        while True:
            with self._state_lock:
                if verbose:
                    print(f"Waiting for mode {mode} current mode {self._control_mode} {mode_t0}")
                if self._control_mode == mode and mode_t0 is None:
                    mode_t0 = timeit.default_timer()
                elif self._control_mode != mode:
                    mode_t0 = None
            # Make sure we are in the mode for at least time_required seconds
            # This is to handle network delays
            if mode_t0 is not None and timeit.default_timer() - mode_t0 > time_required:
                break
            if resend_action is not None:
                self.send_socket.send_pyobj(resend_action)
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                raise RuntimeError(f"Timeout waiting for mode {mode}: {t1 - t0} seconds")

        assert self._control_mode == mode
        return True

    def _wait_for_base_motion(
        self,
        block_id: int,
        verbose: bool = False,
        timeout: float = 10.0,
        moving_threshold: Optional[float] = None,
        angle_threshold: Optional[float] = None,
        min_steps_not_moving: Optional[int] = 1,
        goal_angle: Optional[float] = None,
        goal_angle_threshold: Optional[float] = 0.15,
        resend_action: Optional[dict] = None,
    ) -> None:
        """Wait for the navigation action to finish.

        Args:
            block_id(int): The unique, tracked integer id of the action to wait for
            verbose(bool): Whether to print out debug information
            timeout(float): How long to wait for the action to finish
            moving_threshold(float): How far the robot must move to be considered moving
            angle_threshold(float): How far the robot must rotate to be considered moving
            min_steps_not_moving(int): How many steps the robot must not move for to be considered stopped
            goal_angle(float): The goal angle to reach
            goal_angle_threshold(float): The threshold for the goal angle
            resend_action(dict): The action to resend if the robot is not moving. If none, do not resend.
        """
        print("=" * 20, f"Waiting for {block_id} at goal", "=" * 20)
        last_pos = None
        last_ang = None
        last_obs_t = None
        not_moving_count = 0
        if moving_threshold is None:
            moving_threshold = self._moving_threshold
        if angle_threshold is None:
            angle_threshold = self._angle_threshold
        if min_steps_not_moving is None:
            min_steps_not_moving = self._min_steps_not_moving
        t0 = timeit.default_timer()
        close_to_goal = False

        while True:

            # Minor delay at the end - give it time to get new messages
            time.sleep(0.01)

            with self._state_lock:
                if self._state is None:
                    print("waiting for obs")
                    continue

            xyt = self.get_base_pose()
            pos = xyt[:2]
            ang = xyt[2]
            obs_t = timeit.default_timer()

            if not self.at_goal():
                t0 = timeit.default_timer()
                continue

            moved_dist = np.linalg.norm(pos - last_pos) if last_pos is not None else float("inf")
            angle_dist = angle_difference(ang, last_ang) if last_ang is not None else float("inf")
            if goal_angle is not None:
                angle_dist_to_goal = angle_difference(ang, goal_angle)
                at_goal = angle_dist_to_goal < goal_angle_threshold
            else:
                at_goal = True

            moved_speed = (
                moved_dist / (obs_t - last_obs_t) if last_obs_t is not None else float("inf")
            )
            angle_speed = (
                angle_dist / (obs_t - last_obs_t) if last_obs_t is not None else float("inf")
            )

            not_moving = (
                last_pos is not None
                and moved_speed < moving_threshold
                and angle_speed < angle_threshold
            )
            if not_moving:
                not_moving_count += 1
            else:
                not_moving_count = 0

            # Check if we are at the goal
            # If we are at the goal, we can stop if we are not moving
            last_pos = pos
            last_ang = ang
            last_obs_t = obs_t
            close_to_goal = at_goal
            if verbose:
                print(
                    f"Waiting for step={block_id} {self._last_step} prev={self._last_step} at {pos} moved {moved_dist:0.04f} angle {angle_dist:0.04f} not_moving {not_moving_count} at_goal {self._state['at_goal']}"
                )
                if goal_angle is not None:
                    print(f"Goal angle {goal_angle} angle dist to goal {angle_dist_to_goal}")
            if self._last_step >= block_id and at_goal and not_moving_count > min_steps_not_moving:
                if verbose:
                    print("---> At goal")
                break

            # Resend the action if we are not moving for some reason and it's been provided
            if resend_action is not None and not close_to_goal:
                # Resend the action
                self.send_socket.send_pyobj(resend_action)

            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                print(f"Timeout waiting for block with step id = {block_id}")
                break
                # raise RuntimeError(f"Timeout waiting for block with step id = {block_id}")

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._control_mode == "manipulation"

    def in_navigation_mode(self) -> bool:
        """Returns true if we are navigating (robot head forward, velocity control on)"""
        return self._control_mode == "navigation"

    def last_motion_failed(self) -> bool:
        """Override this if you want to check to see if a particular motion failed, e.g. it was not reachable and we don't know why."""
        return False

    def get_robot_model(self):
        """return a model of the robot for planning"""
        return self._robot_model

    def _update_obs(self, obs):
        """Update observation internally with lock"""
        with self._obs_lock:
            self._obs = obs
            self._last_step = obs["step"]
            if self._iter <= 0:
                self._iter = max(self._last_step, self._iter)

    def _update_pose_graph(self, obs):
        """Update internal pose graph"""
        with self._obs_lock:
            self._pose_graph = obs["pose_graph"]

    def _update_state(self, state: dict) -> None:
        """Update state internally with lock. This is expected to be much more responsive than using full observations, which should be reserved for higher level control.

        Args:
            state (dict): state message from the robot
        """
        with self._state_lock:
            self._state = state
            self._control_mode = state["control_mode"]
            self._at_goal = state["at_goal"]

    def at_goal(self) -> bool:
        """Check if the robot is at the goal.

        Returns:
            at_goal (bool): whether the robot is at the goal
        """
        with self._state_lock:
            if self._state is None:
                return False
            return self._state["at_goal"]

    def save_map(self, filename: str):
        """Save the current map to a file.

        Args:
            filename (str): the filename to save the map to
        """
        next_action = {"save_map": filename}
        self.send_action(next_action)

    def load_map(self, filename: str):
        """Load a map from a file.

        Args:
            filename (str): the filename to load the map from
        """
        next_action = {"load_map": filename}
        self.send_action(next_action)

    def get_observation(self):
        """Get the current observation. This uses the FULL observation track. Expected to be syncd with RGBD."""
        with self._obs_lock:
            if self._obs is None:
                return None
            observation = Observations(
                gps=self._obs["gps"],
                compass=self._obs["compass"],
                rgb=self._obs["rgb"],
                depth=self._obs["depth"],
                xyz=self._obs["xyz"],
                lidar_points=self._obs["lidar_points"],
                lidar_timestamp=self._obs["lidar_timestamp"],
            )
            observation.joint = self._obs.get("joint", None)
            observation.ee_pose = self._obs.get("ee_pose", None)
            observation.camera_K = self._obs.get("camera_K", None)
            observation.camera_pose = self._obs.get("camera_pose", None)
            observation.seq_id = self._seq_id
        return observation

    def get_images(self, compute_xyz=False):
        obs = self.get_observation()
        if compute_xyz:
            return obs.rgb, obs.depth, obs.xyz
        else:
            return obs.rgb, obs.depth

    def get_camera_K(self):
        obs = self.get_observation()
        return obs.camera_K

    def get_head_pose(self):
        obs = self.get_observation()
        return obs.camera_pose

    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        final_timeout: float = 10.0,
        relative: bool = False,
        blocking: bool = False,
    ):
        """Execute a multi-step trajectory; this is always blocking since it waits to reach each one in turn."""

        if isinstance(trajectory, PlanResult):
            trajectory = [pt.state for pt in trajectory.trajectory]

        for i, pt in enumerate(trajectory):
            assert (
                len(pt) == 3 or len(pt) == 2
            ), "base trajectory needs to be 2-3 dimensions: x, y, and (optionally) theta"
            # just_xy = len(pt) == 2
            # self.navigate_to(pt, relative, position_only=just_xy, blocking=False)
            last_waypoint = i == len(trajectory) - 1
            self.navigate_to(
                pt,
                relative,
                blocking=last_waypoint,
                timeout=final_timeout if last_waypoint else per_waypoint_timeout,
            )
            if not last_waypoint:
                self.wait_for_waypoint(
                    pt,
                    pos_err_threshold=pos_err_threshold,
                    rot_err_threshold=rot_err_threshold,
                    rate=spin_rate,
                    verbose=verbose,
                    timeout=per_waypoint_timeout,
                )

    def wait_for_waypoint(
        self,
        xyt: np.ndarray,
        rate: int = 10,
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        verbose: bool = False,
        timeout: float = 20.0,
    ) -> bool:
        """Wait until the robot has reached a configuration... but only roughly. Used for trajectory execution.

        Parameters:
            xyt: se(2) base pose in world coordinates to go to
            rate: rate at which we should check to see if done
            pos_err_threshold: how far robot can be for this waypoint
            verbose: prints extra info out
            timeout: aborts at this point

        Returns:
            success: did we reach waypoint in time"""
        _delay = 1.0 / rate
        xy = xyt[:2]
        if verbose:
            print(f"Waiting for {xyt}, threshold = {pos_err_threshold}")
        # Save start time for exiting trajectory loop
        t0 = timeit.default_timer()
        while not self._finish:
            # Loop until we get there (or time out)
            t1 = timeit.default_timer()
            curr = self.get_base_pose()
            pos_err = np.linalg.norm(xy - curr[:2])
            rot_err = np.abs(angle_difference(curr[-1], xyt[2]))
            # TODO: code for debugging slower rotations
            # if pos_err < pos_err_threshold and rot_err > rot_err_threshold:
            #     print(f"{curr[-1]}, {xyt[2]}, {rot_err}")
            if verbose:
                logger.info(f"- {curr=} target {xyt=} {pos_err=} {rot_err=}")
            if pos_err < pos_err_threshold and rot_err < rot_err_threshold:
                # We reached the goal position
                return True
            t2 = timeit.default_timer()
            dt = t2 - t1
            if t2 - t0 > timeout:
                logger.warning(
                    "[WAIT FOR WAYPOINT] WARNING! Could not reach goal in time: "
                    + str(xyt)
                    + " "
                    + str(curr)
                )
                return False
            time.sleep(max(0, _delay - (dt)))
        return False

    def set_base_velocity(self, forward: float, rotational: float) -> None:
        """Set the velocity of the robot base.

        Args:
            forward (float): forward velocity
            rotational (float): rotational velocity
        """
        next_action = {"base_velocity": {"v": forward, "w": rotational}}
        self.send_action(next_action)

    def send_action(
        self,
        next_action: Dict[str, Any],
        timeout: float = 5.0,
        verbose: bool = False,
        force_resend: bool = False,
    ) -> Dict[str, Any]:
        """Send the next action to the robot. Increment the step counter and wait for the action to finish if it is blocking.

        Args:
            next_action (dict): the action to send
            timeout (float): how long to wait for the action to finish
            verbose (bool): whether to print out debug information
            force_resend (bool): whether to resend the action

        Returns:
            dict: copy of the action that was sent to the robot.
        """
        if verbose:
            logger.info("-> sending", next_action)
        blocking = False
        block_id = None
        with self._act_lock:

            # Get blocking
            blocking = next_action.get("nav_blocking", False)
            block_id = self._iter
            # Send it
            next_action["step"] = block_id
            self._iter += 1

            # TODO: fix all of this - why do we need to do this?
            # print("SENDING THIS ACTION:", next_action)
            self.send_socket.send_pyobj(next_action)

            if self._resend_all_actions or force_resend:
                time.sleep(0.01)

                logger.debug("RESENDING THIS ACTION:", next_action)
                self.send_socket.send_pyobj(next_action)

            # For tracking goal
            if "xyt" in next_action:
                goal_angle = next_action["xyt"][2]
            else:
                goal_angle = None

            # Empty it out for the next one
            current_action = next_action

        # Make sure we had time to read
        # time.sleep(0.1)
        if blocking:
            # Wait for the command to finish
            self._wait_for_base_motion(
                block_id,
                goal_angle=goal_angle,
                verbose=verbose,
                timeout=timeout,
                # resend_action=current_action,
            )

        # Returns the current action in case we want to do something with it like resend
        return current_action

    def blocking_spin(self, verbose: bool = False, visualize: bool = False):
        """Listen for incoming observations and update internal state"""
        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()
        camera = None
        shown_point_cloud = visualize

        while not self._finish:

            output = self.recv_socket.recv_pyobj()
            if output is None:
                continue

            self._seq_id += 1
            output["rgb"] = compression.from_jpg(output["rgb"])
            compressed_depth = output["depth"]
            depth = compression.from_jp2(compressed_depth) / 1000
            output["depth"] = depth

            if camera is None:
                camera = Camera.from_K(
                    output["camera_K"], output["rgb_height"], output["rgb_width"]
                )

            output["xyz"] = camera.depth_to_xyz(output["depth"])

            if visualize and not shown_point_cloud:
                show_point_cloud(output["xyz"], output["rgb"] / 255.0, orig=np.zeros(3))
                shown_point_cloud = True

            self._update_obs(output)
            self._update_pose_graph(output)

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose:
                print("Control mode:", self._control_mode)
                print(f"time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}")
            t0 = timeit.default_timer()

    def update_servo(self, message):
        """Servo messages"""
        if message is None or self._state is None:
            return

        # color_image = compression.from_webp(message["ee_cam/color_image"])
        if "ee_cam/color_image" in message:
            color_image = compression.from_jpg(message["ee_cam/color_image"])
            depth_image = compression.from_jp2(message["ee_cam/depth_image"])
            depth_image = depth_image / 1000
        else:
            color_image = None
            depth_image = None
            image_scaling = None

        # Get head information from the message as well
        head_color_image = compression.from_jpg(message["head_cam/color_image"])
        head_depth_image = compression.from_jp2(message["head_cam/depth_image"]) / 1000
        head_image_scaling = message["head_cam/image_scaling"]
        joint = message["robot/config"]
        with self._servo_lock and self._state_lock:
            observation = Observations(
                gps=self._state["base_pose"][:2],
                compass=self._state["base_pose"][2],
                rgb=head_color_image,
                depth=head_depth_image,
                xyz=None,
                ee_rgb=color_image,
                ee_depth=depth_image,
                ee_xyz=None,
                joint=joint,
            )

            # We may not have the camera information yet
            # Some robots do not have the d405
            if "ee_cam/depth_camera_K" in message:
                observation.ee_camera_K = message["ee_cam/depth_camera_K"]
                observation.ee_camera_pose = message["ee_cam/pose"]
                observation.ee_depth_scaling = message["ee_cam/image_scaling"]

            observation.ee_pose = message["ee/pose"]
            observation.depth_scaling = message["head_cam/depth_scaling"]
            observation.camera_K = message["head_cam/depth_camera_K"]
            observation.camera_pose = message["head_cam/pose"]
            if "is_simulation" in message:
                observation.is_simulation = message["is_simulation"]
            else:
                observation.is_simulation = False
            self._servo = observation

    def get_servo_observation(self):
        """Get the current servo observation.

        Returns:
            Observations: the current servo observation
        """
        with self._servo_lock:
            return self._servo

    def blocking_spin_servo(self, verbose: bool = False):
        """Listen for servo messages coming from the robot, i.e. low res images for ML state. This is intended to be run in a separate thread.

        Args:
            verbose (bool): whether to print out debug information
        """
        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()
        while not self._finish:
            t1 = timeit.default_timer()
            dt = t1 - t0
            output = self.recv_servo_socket.recv_pyobj()
            self.update_servo(output)
            sum_time += dt
            steps += 1
            if verbose and steps % self.num_state_report_steps == 1:
                print(
                    f"[SERVO] time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
                )
            t0 = timeit.default_timer()

    @property
    def running(self) -> bool:
        """Is the client running? Best practice is to check this during while loops.

        Returns:
            bool: whether the client is running
        """
        return not self._finish

    def is_running(self) -> bool:
        """Is the client running? Best practice is to check this during while loops.

        Returns:
            bool: whether the client is running
        """
        return not self._finish

    def say(self, text: str):
        """Send a text message to the robot to say. Will be spoken by the robot's text-to-speech system asynchronously."""
        next_action = {"say": text}
        self.send_action(next_action)

    def blocking_spin_state(self, verbose: bool = False):
        """Listen for incoming observations and update internal state"""

        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()

        while not self._finish:
            output = self.recv_state_socket.recv_pyobj()
            self._update_state(output)

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose and steps % self.num_state_report_steps == 1:
                print("[STATE] Control mode:", self._control_mode)
                print(
                    f"[STATE] time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
                )
            t0 = timeit.default_timer()

    def blocking_spin_rerun(self) -> None:
        """Use the rerun server so that we can visualize what is going on as the robot takes actions in the world."""
        while not self._finish:
            self._rerun.step(self._obs, self._servo)

    @property
    def is_homed(self) -> bool:
        """Is the robot homed?

        Returns:
            bool: whether the robot is homed
        """
        # This is not really thread safe
        with self._state_lock:
            return self._state is not None and self._state["is_homed"]

    @property
    def is_runstopped(self) -> bool:
        """Is the robot runstopped?

        Returns:
            bool: whether the robot is runstopped
        """
        with self._state_lock:
            return self._state is not None and self._state["is_runstopped"]

    def start(self) -> bool:
        """Start running blocking thread in a separate thread. This will wait for observations to come in and update internal state.

        Returns:
            bool: whether the client was started successfully
        """
        if self._started:
            # Already started
            return True

        self._thread = threading.Thread(target=self.blocking_spin)
        self._state_thread = threading.Thread(target=self.blocking_spin_state)
        self._servo_thread = threading.Thread(target=self.blocking_spin_servo)
        if self._rerun:
            self._rerun_thread = threading.Thread(target=self.blocking_spin_rerun)  # type: ignore
        self._finish = False
        self._thread.start()
        self._state_thread.start()
        self._servo_thread.start()
        if self._rerun:
            self._rerun_thread.start()

        t0 = timeit.default_timer()
        while self._obs is None or self._state is None or self._servo is None:
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > 10.0:
                logger.error(
                    colored(
                        "Timeout waiting for observations; are you connected to the robot? Check the network.",
                        "red",
                    )
                )
                logger.info(
                    "Try making sure that the server on the robot is publishing, and that you can ping the robot IP address."
                )
                logger.info("Robot IP:", self.send_address)
                return False

        # Separately wait for state messages
        while self._state is None:
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > 10.0:
                logger.error(
                    colored(
                        "Timeout waiting for state information; are you connected to the robot? Check the network.",
                        "red",
                    )
                )

        if not self.is_homed:
            self.stop()
            raise RuntimeError(
                "Robot is not homed; please home the robot before running. You can do so by shutting down the server and running ./stretch_robot_home.py on the robot."
            )
        if self.is_runstopped:
            self.stop()
            raise RuntimeError(
                "Robot is runstopped; please release the runstop before running. You can do so by pressing and briefly holding the runstop button on the robot."
            )

        self._started = True
        return True

    def __del__(self):
        """Destructor to make sure we stop the client when it is deleted"""
        self.stop()

    def stop(self):
        """Stop the client and close all sockets"""
        self._finish = True
        if self._thread is not None:
            self._thread.join()
        if self._state_thread is not None:
            self._state_thread.join()
        if self._servo_thread is not None:
            self._servo_thread.join()
        if self._rerun_thread is not None:
            self._rerun_thread.join()

        # Close the sockets and context
        self.recv_socket.close()
        self.recv_state_socket.close()
        self.recv_servo_socket.close()
        self.send_socket.close()
        self.context.term()


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="192.168.1.15")
def main(
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "192.168.1.15",
):
    client = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
    )
    client.blocking_spin(verbose=True, visualize=False)


if __name__ == "__main__":
    main()
