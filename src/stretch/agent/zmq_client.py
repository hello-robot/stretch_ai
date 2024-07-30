# (c) 2024 chris paxton under MIT license

import threading
import time
import timeit
from threading import Lock
from typing import List, Optional

import click
import cv2
import numpy as np
import zmq
from termcolor import colored

import stretch.motion.constants as constants
import stretch.motion.conversions as conversions
import stretch.utils.compression as compression
import stretch.utils.logger as logger
from stretch.core.interfaces import ContinuousNavigationAction, Observations
from stretch.core.parameters import Parameters, get_parameters
from stretch.core.robot import RobotClient
from stretch.motion import PlanResult, RobotModel
from stretch.motion.kinematics import HelloStretchIdx, HelloStretchKinematics
from stretch.utils.geometry import angle_difference
from stretch.utils.image import Camera
from stretch.utils.network import lookup_address
from stretch.utils.point_cloud import show_point_cloud

# TODO: debug code - remove later if necessary
# import faulthandler
# faulthandler.enable()


class HomeRobotZmqClient(RobotClient):

    update_base_pose_from_full_obs: bool = False
    num_state_report_steps: int = 10000

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

    def __init__(
        self,
        robot_ip: str = "",
        recv_port: int = 4401,
        send_port: int = 4402,
        recv_state_port: int = 4403,
        recv_servo_port: int = 4404,
        parameters: Parameters = None,
        use_remote_computer: bool = True,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
        start_immediately: bool = True,
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

        # Variables we set here should not change
        self._iter = 0  # Tracks number of actions set, never reset this
        self._seq_id = 0  # Number of messages we received
        self._started = False

        if parameters is None:
            parameters = get_parameters("default_planner.yaml")
        self._parameters = parameters
        self._moving_threshold = parameters["motion"]["moving_threshold"]
        self._angle_threshold = parameters["motion"]["angle_threshold"]
        self._min_steps_not_moving = parameters["motion"]["min_steps_not_moving"]

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

        # SEnd actions back to the robot for execution
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

        if start_immediately:
            self.start()

    def get_joint_state(self, timeout: float = 5.0) -> np.ndarray:
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

    def get_base_pose(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current pose of the base"""
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

    def robot_to(self, joint_angles: np.ndarray, blocking: bool = False, timeout: float = 10.0):
        """Move the robot to a particular joint configuration."""
        with self._act_lock:
            self._next_action["joint"] = joint_angles
            self._next_action["manip_blocking"] = blocking
        self.send_action(timeout=timeout)

    def head_to(
        self, head_pan: float, head_tilt: float, blocking: bool = False, timeout: float = 10.0
    ):
        """Move the head to a particular configuration."""
        with self._act_lock:
            self._next_action["head_to"] = [float(head_pan), float(head_tilt)]
        self.send_action(timeout=timeout)

        if blocking:
            whole_body_q = np.zeros(self._robot_model.dof, dtype=np.float32)
            whole_body_q[HelloStretchIdx.HEAD_PAN] = float(head_pan)
            whole_body_q[HelloStretchIdx.HEAD_TILT] = float(head_tilt)
            self._wait_for_head(whole_body_q)

    def arm_to(
        self,
        joint_angles: Optional[np.ndarray] = None,
        gripper: float = None,
        blocking: bool = False,
        timeout: float = 10.0,
        verbose: bool = False,
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
            joint_angles = np.zeros(self._robot_model.dof)
        elif len(joint_angles) > 6:
            print(
                "[WARNING] arm_to: attempting to convert from full robot state to 6dof manipulation state."
            )
            joint_angles = self._robot_model.config_to_manip_command(joint_angles)

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
        # Now send
        with self._act_lock:
            self._next_action["joint"] = joint_angles
            if gripper:
                self._next_action["gripper"] = gripper
            self._next_action["manip_blocking"] = blocking

        # Blocking is handled in here
        self.send_action()

        # Handle blocking
        steps = 0
        if blocking:
            t0 = timeit.default_timer()
            while not self._finish:

                if steps % 10 == 9:
                    # Resend the action until we get there
                    with self._act_lock:
                        self._next_action["joint"] = joint_angles
                        self._next_action["manip_blocking"] = blocking
                    self.send_action()

                joint_state = self.get_joint_state()
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
                if (
                    (arm_diff < 0.05)
                    and (lift_diff < 0.05)
                    and (base_x_diff < 0.05)
                    and (wrist_roll_diff < 0.05)
                    and (wrist_pitch_diff < 0.05)
                    and (wrist_yaw_diff < 0.05)
                ):
                    return True
                else:
                    if verbose:
                        print(
                            f"{arm_diff=}, {lift_diff=}, {base_x_diff=}, {wrist_roll_diff=}, {wrist_pitch_diff=}, {wrist_yaw_diff=}"
                        )
                time.sleep(0.01)

                # TODO: Is this necessary? If not, we should just delete this commented-out code block.
                # Resend the action
                # self._next_action["joint"] = joint_angles
                # self.send_action()

                t1 = timeit.default_timer()
                if t1 - t0 > timeout:
                    print("[ZMQ CLIENT] Timeout waiting for arm to move")
                    break
                steps += 1
            return False
        return True

    def navigate_to(
        self, xyt: ContinuousNavigationAction, relative=False, blocking=False, timeout: float = 10.0
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        if isinstance(xyt, ContinuousNavigationAction):
            xyt = xyt.xyt
        assert len(xyt) == 3, "xyt must be a vector of size 3"
        with self._act_lock:
            self._next_action["xyt"] = xyt
            self._next_action["nav_relative"] = relative
            self._next_action["nav_blocking"] = blocking
        self.send_action(timeout=timeout)

    def reset(self):
        """Reset everything in the robot's internal state"""
        self._control_mode = None
        self._next_action = dict()
        self._obs = None  # Full observation includes high res images and camera pose, no EE camera
        self._state = None  # Low level state includes joint angles and base XYT
        self._servo = None  # Visual servoing state includes smaller images
        self._thread = None
        self._finish = False
        self._last_step = -1

    def open_gripper(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """Open the gripper based on hard-coded presets."""
        gripper_target = self._robot_model.GRIPPER_OPEN
        print("[ZMQ CLIENT] Opening gripper to", gripper_target)
        self.gripper_to(gripper_target, blocking=False)
        if blocking:
            t0 = timeit.default_timer()
            while not self._finish:
                joint_state = self.get_joint_state()
                if joint_state is None:
                    continue
                gripper_err = np.abs(joint_state[HelloStretchIdx.GRIPPER] - gripper_target)
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

    def close_gripper(self, blocking: bool = True, timeout: float = 10.0) -> bool:
        """Close the gripper based on hard-coded presets."""
        gripper_target = self._robot_model.GRIPPER_CLOSED
        print("[ZMQ CLIENT] Closing gripper to", gripper_target)
        self.gripper_to(gripper_target, blocking=False)
        if blocking:
            t0 = timeit.default_timer()
            while not self._finish:
                joint_state = self.get_joint_state()
                if joint_state is None:
                    continue
                gripper_err = np.abs(joint_state[HelloStretchIdx.GRIPPER] - gripper_target)
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
        with self._act_lock:
            self._next_action["gripper"] = target
            self._next_action["gripper_blocking"] = blocking
        self.send_action()
        if blocking:
            time.sleep(2.0)

    def switch_to_navigation_mode(self):
        """Velocity control of the robot base."""
        with self._act_lock:
            self._next_action["control_mode"] = "navigation"
        self.send_action()
        self._wait_for_mode("navigation")
        assert self.in_navigation_mode()

    def in_navigation_mode(self) -> bool:
        """Returns true if we are navigating (robot head forward, velocity control on)"""
        return self._control_mode == "navigation"

    def in_manipulation_mode(self) -> bool:
        return self._control_mode == "manipulation"

    def switch_to_manipulation_mode(self):
        with self._act_lock:
            self._next_action["control_mode"] = "manipulation"
        self.send_action()
        time.sleep(0.1)
        self._wait_for_mode("manipulation")
        assert self.in_manipulation_mode()

    def move_to_nav_posture(self):
        with self._act_lock:
            self._next_action["posture"] = "navigation"
        self.send_action()
        self._wait_for_head(constants.STRETCH_NAVIGATION_Q, resend_action={"posture": "navigation"})
        self._wait_for_mode("navigation")
        assert self.in_navigation_mode()

    def move_to_manip_posture(self):
        with self._act_lock:
            self._next_action["posture"] = "manipulation"
        self.send_action()
        time.sleep(0.1)
        self._wait_for_head(constants.STRETCH_PREGRASP_Q, resend_action={"posture": "manipulation"})
        self._wait_for_mode("manipulation")
        assert self.in_manipulation_mode()

    def _wait_for_head(
        self,
        q: np.ndarray,
        timeout: float = 10.0,
        resend_action: Optional[dict] = None,
        verbose: bool = True,
    ) -> None:
        """Wait for the head to move to a particular configuration."""
        t0 = timeit.default_timer()
        while True:
            joint_state = self.get_joint_state()
            if joint_state is None:
                continue
            pan_err = np.abs(joint_state[HelloStretchIdx.HEAD_PAN] - q[HelloStretchIdx.HEAD_PAN])
            tilt_err = np.abs(joint_state[HelloStretchIdx.HEAD_TILT] - q[HelloStretchIdx.HEAD_TILT])
            if verbose:
                print("Waiting for head to move", pan_err, tilt_err)
            if pan_err < 0.1 and tilt_err < 0.1:
                break
            elif resend_action is not None:
                self.send_socket.send_pyobj(resend_action)
            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                print("Timeout waiting for head to move")
                break
            time.sleep(0.01)
        # Tiny pause after head rotation
        time.sleep(0.5)

    def _wait_for_mode(self, mode, verbose: bool = False, timeout: float = 20.0):
        t0 = timeit.default_timer()
        while True:
            with self._state_lock:
                if verbose:
                    print(f"Waiting for mode {mode} current mode {self._control_mode}")
                if self._control_mode == mode:
                    break
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                raise RuntimeError(f"Timeout waiting for mode {mode}: {t1 - t0} seconds")
        assert self._control_mode == mode

    def _wait_for_action(
        self,
        block_id: int,
        verbose: bool = False,
        timeout: float = 10.0,
        moving_threshold: Optional[float] = None,
        angle_threshold: Optional[float] = None,
        min_steps_not_moving: Optional[int] = 1,
        goal_angle: Optional[float] = None,
        goal_angle_threshold: Optional[float] = 0.1,
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
            with self._obs_lock:
                if self._obs is None:
                    continue
            xyt = self.get_base_pose()
            pos = xyt[:2]
            ang = xyt[2]

            if not self.at_goal():
                t0 = timeit.default_timer()
                continue

            moved_dist = np.linalg.norm(pos - last_pos) if last_pos is not None else 0
            angle_dist = angle_difference(ang, last_ang) if last_ang is not None else 0
            if goal_angle is not None:
                angle_dist_to_goal = angle_difference(ang, goal_angle)
                at_goal = angle_dist_to_goal < goal_angle_threshold
            else:
                at_goal = True
            not_moving = (
                last_pos is not None
                and moved_dist < moving_threshold
                and angle_dist < angle_threshold
            )
            if not_moving:
                not_moving_count += 1
            else:
                not_moving_count = 0
            last_pos = pos
            last_ang = ang
            close_to_goal = at_goal
            if verbose:
                print(
                    f"Waiting for step={block_id} {self._last_step} prev={self._last_step} at {pos} moved {moved_dist:0.04f} angle {angle_dist:0.04f} not_moving {not_moving_count} at_goal {self._obs['at_goal']}"
                )
                if goal_angle is not None:
                    print(f"Goal angle {goal_angle} angle dist to goal {angle_dist_to_goal}")
            if self._last_step >= block_id and at_goal and not_moving_count > min_steps_not_moving:
                break

            # Resend the action if we are not moving for some reason and it's been provided
            if resend_action is not None and not close_to_goal:
                # Resend the action
                self.send_socket.send_pyobj(resend_action)

            # Minor delay at the end - give it time to get new messages
            time.sleep(0.01)
            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                raise RuntimeError(f"Timeout waiting for block with step id = {block_id}")

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._control_mode == "manipulation"

    def in_navigation_mode(self) -> bool:
        """Is the robot to move around"""
        return self._control_mode == "navigation"

    def last_motion_failed(self) -> bool:
        """Override this if you want to check to see if a particular motion failed, e.g. it was not reachable and we don't know why."""
        return False

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        return self._robot_model

    def _update_obs(self, obs):
        """Update observation internally with lock"""
        with self._obs_lock:
            self._obs = obs
            self._last_step = obs["step"]
            if self._iter <= 0:
                self._iter = self._last_step

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
            )
            observation.joint = self._obs.get("joint", None)
            observation.ee_pose = self._obs.get("ee_pose", None)
            observation.camera_K = self._obs.get("camera_K", None)
            observation.camera_pose = self._obs.get("camera_pose", None)
            observation.seq_id = self._seq_id
        return observation

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
                    "[WAIT FOR WAYPOINT] WARNING! Could not reach goal in time: " + str(xyt)
                )
                return False
            time.sleep(max(0, _delay - (dt)))
        return False

    def send_action(self, timeout: float = 10.0, verbose: bool = False) -> None:
        """Send the next action to the robot"""
        if verbose:
            print("-> sending", self._next_action)
        blocking = False
        block_id = None
        with self._act_lock:
            blocking = self._next_action.get("nav_blocking", False)
            block_id = self._iter
            # Send it
            self._next_action["step"] = block_id
            self._iter += 1
            self.send_socket.send_pyobj(self._next_action)

            # For tracking goal
            if "xyt" in self._next_action:
                goal_angle = self._next_action["xyt"][2]
            else:
                goal_angle = None

            # Empty it out for the next one
            current_action = self._next_action
            self._next_action = dict()

        # Make sure we had time to read
        time.sleep(0.1)
        if blocking:
            # Wait for the command to finish
            self._wait_for_action(
                block_id,
                goal_angle=goal_angle,
                verbose=verbose,
                timeout=timeout,
                # resend_action=current_action,
            )
            time.sleep(0.1)

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
            depth = compression.from_jp2(compressed_depth)
            output["depth"] = depth / 1000.0

            if camera is None:
                camera = Camera.from_K(
                    output["camera_K"], output["rgb_height"], output["rgb_width"]
                )

            output["xyz"] = camera.depth_to_xyz(output["depth"])

            if visualize and not shown_point_cloud:
                show_point_cloud(output["xyz"], output["rgb"] / 255.0, orig=np.zeros(3))
                shown_point_cloud = True

            self._update_obs(output)
            # with self._act_lock:
            #    if len(self._next_action) > 0:

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
        color_image = compression.from_jpg(message["ee_cam/color_image"])
        depth_image = compression.from_jp2(message["ee_cam/depth_image"])
        image_scaling = message["ee_cam/image_scaling"]

        # Get head information from the message as well
        head_color_image = compression.from_jpg(message["head_cam/color_image"])
        head_depth_image = compression.from_jp2(message["head_cam/depth_image"])
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
            observation.camera_K = message["head_cam/depth_camera_K"]
            observation.ee_camera_K = message["ee_cam/depth_camera_K"]
            observation.camera_pose = message["head_cam/pose"]
            observation.ee_camera_pose = message["ee_cam/pose"]
            observation.depth_scaling = message["head_cam/depth_scaling"]
            observation.ee_depth_scaling = message["ee_cam/image_scaling"]
            self._servo = observation

    def get_servo_observation(self):
        """Get the current servo observation"""
        with self._servo_lock:
            return self._servo

    def blocking_spin_servo(self, verbose: bool = False):
        """Listen for servo messages coming from the robot, i.e. low res images for ML state"""
        sum_time = 0
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

    def blocking_spin_state(self, verbose: bool = False):
        """Listen for incoming observations and update internal state"""

        sum_time = 0
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

    def start(self) -> bool:
        """Start running blocking thread in a separate thread"""
        if self._started:
            return True

        self._thread = threading.Thread(target=self.blocking_spin)
        self._state_thread = threading.Thread(target=self.blocking_spin_state)
        self._servo_thread = threading.Thread(target=self.blocking_spin_servo)
        self._finish = False
        self._thread.start()
        self._state_thread.start()
        self._servo_thread.start()

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
        self._started = True
        return True

    def __del__(self):
        self.stop()

    def stop(self):
        self._finish = True
        if self._thread is not None:
            self._thread.join()
        if self._state_thread is not None:
            self._state_thread.join()
        if self._servo_thread is not None:
            self._servo_thread.join()

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
