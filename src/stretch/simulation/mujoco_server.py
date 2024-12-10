#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import random
import threading
import time
import timeit
from typing import Any, Dict, Optional

import click
import numpy as np
from overrides import override
from stretch_mujoco import StretchMujocoSimulator

try:
    from stretch.simulation.robocasa_gen import load_model_from_xml, model_generation_wizard
except ImportError as e:
    from stretch.utils.logger import error

    error("Could not import robocasa!")
    error("Install robosuite and robocasa in order to use model generation wizard.")
    error(f"Error: {e}")

import stretch.motion.constants as constants
import stretch.simulation.utils as sim_utils
import stretch.utils.compression as compression
import stretch.utils.logger as logger
from stretch.core.server import BaseZmqServer
from stretch.motion import HelloStretchIdx
from stretch.motion.control.goto_controller import GotoVelocityController
from stretch.utils.config import get_control_config
from stretch.utils.geometry import pose_global_to_base, xyt_base_to_global, xyt_global_to_base
from stretch.utils.image import scale_camera_matrix

# Maps HelloStretchIdx to actuators
mujoco_actuators = {
    HelloStretchIdx.BASE_X: "base_x_joint",
    HelloStretchIdx.BASE_Y: "base_y_joint",
    HelloStretchIdx.BASE_THETA: "base_theta_joint",
    HelloStretchIdx.LIFT: "lift",
    HelloStretchIdx.ARM: "arm",
    HelloStretchIdx.GRIPPER: "gripper",
    HelloStretchIdx.WRIST_ROLL: "wrist_roll",
    HelloStretchIdx.WRIST_PITCH: "wrist_pitch",
    HelloStretchIdx.WRIST_YAW: "wrist_yaw",
    HelloStretchIdx.HEAD_PAN: "head_pan",
    HelloStretchIdx.HEAD_TILT: "head_tilt",
}

stretch_dof = constants.stretch_degrees_of_freedom

manip_idx = [
    HelloStretchIdx.BASE_X,
    HelloStretchIdx.LIFT,
    HelloStretchIdx.ARM,
    HelloStretchIdx.WRIST_ROLL,
    HelloStretchIdx.WRIST_PITCH,
    HelloStretchIdx.WRIST_YAW,
]


# Constants for the controller
CONTROL_HZ = 20
VEL_THRESHOlD = 0.001
RVEL_THRESHOLD = 0.005


class MujocoZmqServer(BaseZmqServer):
    """Server for Mujoco simulation with ZMQ communication. This allows us to run the Mujoco simulation in the exact same way as we would run a remote ROS server on the robot, including potentially running it on a different machine or on the cloud. It requires:
    - Mujoco installation
    - Stretch_mujoco installation: https://github.com/hello-robot/stretch_mujoco/
    """

    # Do we use a navigation controller command to move the robot back at the start of a Robocasa task?
    _move_back_at_start: bool = False

    hz = CONTROL_HZ
    # How long should the controller report done before we're actually confident that we're done?
    done_t = 0.1

    robocasa_start_offset = np.array([0.5, 0, -np.pi / 2])

    # Print debug messages for control loop
    debug_control_loop = False
    debug_set_goal_pose = False

    def get_body_xyt(self, body_name: str) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""

        # Get mjdata and mjmodel from simulator
        mjdata = self.robot_sim.mjdata
        mjmodel = self.robot_sim.mjmodel

        xyz = mjdata.body(body_name).xpos
        rotation = mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def list_all_bodies(self):
        """Debug function to list all bodies in the simulation."""
        model = self.robot_sim.mjmodel
        data = self.robot_sim.mjdata

        # Loop over all bodies
        for i in range(model.nbody):
            body_name = model.body(i).name
            body_pos = data.body(i).xpos
            body_quat = data.body(i).xquat

            print(f"Body {i}: {body_name}")
            print(f"  Position: {body_pos}")
            print(f"  Orientation (quaternion): {body_quat}")

    def set_robot_position(self, xyt: np.ndarray, relative: bool = False) -> None:
        """Set the robot position in the simulation

        Args:
            xyt (np.ndarray): The desired pose of the robot in world coordinates (x, y, theta).
            relative (bool): If True, the pose is relative to the current pose of the robot.
        """
        return self.set_body_position(xyt, "base_link", relative=relative)

    def set_body_position(self, xyt: np.ndarray, body_name: str, relative: bool = False) -> None:
        """Set the robot position in the simulation.

        Args:
            xyt (np.ndarray): The desired pose of the robot in world coordinates (x, y, theta).
            body_name (str): The name of the body to set the position of.
            relative (bool): If True, the pose is relative to the current pose of the robot.
        """

        # Get mjdata and mjmodel from simulator
        mjdata = self.robot_sim.mjdata
        mjmodel = self.robot_sim.mjmodel

        # Compute absolute goal
        if relative:
            xyt_base = self.get_body_xyt(body_name)
            xyt_goal = xyt_base_to_global(xyt, xyt_base)
        else:
            xyt_goal = xyt

        # Get current position from mjdata
        # TODO: remove debug print
        # body_names = [mjmodel.body(i).name for i in range(mjmodel.nbody)]
        # print(body_names)
        base_pos = mjdata.body("base_link").xpos
        print(f"Current {body_name} position: {base_pos}")
        base_pos[0] = xyt_goal[0]
        base_pos[1] = xyt_goal[1]
        print(f"Setting {body_name} to {base_pos}")
        mjdata.body("base_link").xpos = base_pos

        # Convert the theta into a rotation matrix
        rotation = mjdata.body("base_link").xmat.reshape(3, 3)
        rotation[0, 0] = np.cos(xyt_goal[2])
        rotation[0, 1] = -np.sin(xyt_goal[2])
        rotation[1, 0] = np.sin(xyt_goal[2])
        rotation[1, 1] = np.cos(xyt_goal[2])
        mjdata.body("base_link").xmat = rotation.flatten()

    def reset(self):
        """Reset the robot to the initial state."""
        self.robot_sim.reset_state()

    def __init__(
        self,
        *args,
        scene_path: Optional[str] = None,
        scene_model: Optional[str] = None,
        simulation_rate: int = 200,
        config_name: str = "noplan_velocity_sim",
        objects_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super(MujocoZmqServer, self).__init__(*args, **kwargs)
        # TODO: decide how we want to save scenes, if they should be here in stretch_ai or in stretch_mujoco
        # They should probably stay in stretch mujoco
        if scene_path is None:
            scene_path = sim_utils.get_default_scene_path()
        if scene_model is not None:
            if scene_path is not None:
                logger.warning("Both scene model and scene path provided. Using scene model.")
            self.robot_sim = StretchMujocoSimulator(model=scene_model)
        else:
            self.robot_sim = StretchMujocoSimulator(scene_xml_path=scene_path)
        self.simulation_rate = simulation_rate
        self.objects_info = objects_info

        # Hard coded printout rates
        self.report_steps = 1000
        self.fast_report_steps = 10000
        self.servo_report_steps = 1000

        self._camera_data = None
        self._status = None
        self._initial_xyt = None
        self._manip_xyt = None

        # Controller stuff
        # Is the velocity controller active?
        # TODO: not sure if we want this
        # self.active = False
        # Is it done?
        self.is_done = False
        # Goal set time
        self.goal_set_t: Optional[float] = None
        self.xyt_goal: Optional[np.ndarray] = None
        self._base_controller_at_goal = False
        self.control_mode = "navigation"
        self.controller_finished = True
        self.active = False

        # Control module
        controller_cfg = get_control_config(config_name)
        self.controller = GotoVelocityController(controller_cfg)
        # Update the velocity and acceleration configs from the file
        self.controller.update_velocity_profile(
            controller_cfg.v_max,
            controller_cfg.w_max,
            controller_cfg.acc_lin,
            controller_cfg.acc_ang,
        )

    def set_goal_pose(self, xyt_goal: np.ndarray, relative: bool = False):
        """Set the goal pose for the robot. The controller will then try to reach this goal pose.

        Args:
            xyt_goal (np.ndarray): Goal pose for the robot in world coordinates (x, y, theta).
            me/cpaxton/miniforge3/envs/stretch_ai_0.0.10/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
                self.run()
                  File "/home/cpaxton/miniforge3/envs/stretch_ai_0.0.10/lib/python3.10/threading.py", line 953, in run
                      self._target(*self._args, **self._kwargs)

        """
        assert len(xyt_goal) == 3, "Goal pose should be of size 3 (x, y, theta)"

        # Compute absolute goal
        if relative:
            xyt_base = self.get_base_pose()
            xyt_goal = xyt_base_to_global(xyt_goal, xyt_base)
        else:
            xyt_goal = xyt_goal

        if self.debug_control_loop or self.debug_set_goal_pose:
            print("-" * 20)
            print("Control loop callback: ", self.active, self.xyt_goal)
            print("Currently at:", self.get_base_pose())
            print("Setting goal to:", xyt_goal)
            print("Passed goal was: ", xyt_goal)
            print("Relative: ", relative)
            print("-" * 20)

        self.controller.update_goal(xyt_goal)
        self.xyt_goal = self.controller.xyt_goal
        self.active = True

        self.is_done = False
        self.goal_set_t = timeit.default_timer()
        self.controller_finished = False
        self._base_controller_at_goal = False

    def _control_loop_thread(self):
        """Control loop thread for the velocity controller"""
        while self.is_running():
            self.control_loop_callback()
            time.sleep(1 / self.hz)

    def control_loop_callback(self):
        """Actual controller timer callback"""

        if self._status is None:
            vel_odom = [0, 0]
        else:
            vel_odom = self._status["base"]["x_vel"], self._status["base"]["theta_vel"]

        if self.debug_control_loop:
            print("Control loop callback: ", self.active, self.xyt_goal, vel_odom)

        base_xyt = self.get_base_pose()
        if base_xyt is None:
            return

        self.controller.update_pose_feedback(base_xyt)

        if self.active and self.xyt_goal is not None:
            # Compute control
            self.is_done = False
            v_cmd, w_cmd = self.controller.compute_control()
            done = self.controller.is_done()

            # self.get_logger().info(f"veclocities {v_cmd} and {w_cmd}")
            # Compute timeout
            time_since_goal_set = timeit.default_timer() - self.goal_set_t
            if self.controller.timeout(time_since_goal_set):
                done = True
                v_cmd, w_cmd = 0, 0

            # Check if actually done (velocity = 0)
            if done and vel_odom is not None:
                if vel_odom[0] < VEL_THRESHOlD and vel_odom[1] < RVEL_THRESHOLD:
                    if not self.controller_finished:
                        self.controller_finished = True
                        self.done_since = timeit.default_timer()
                    elif (
                        self.controller_finished
                        and (timeit.default_timer() - self.done_since) > self.done_t
                    ):
                        self.is_done = True
                else:
                    self.controller_finished = False
                    self.done_since = timeit.default_timer()

            # Command robot
            if self.debug_control_loop:
                print(f"Commanding robot with {v_cmd} and {w_cmd}")
            self.robot_sim.set_base_velocity(v_linear=v_cmd, omega=w_cmd)
            self._base_controller_at_goal = self.controller_finished and self.is_done

            if self.is_done:
                self.active = False
                self.xyt_goal = None

    def base_controller_at_goal(self):
        """Check if the base controller is at goal."""
        return self._base_controller_at_goal

    def get_joint_state(self):
        """Get the joint state of the robot."""
        status = self._status

        positions = np.zeros(constants.stretch_degrees_of_freedom)
        velocities = np.zeros(constants.stretch_degrees_of_freedom)
        efforts = np.zeros(constants.stretch_degrees_of_freedom)

        if status is None:
            return positions, velocities, efforts

        # Lift joint
        positions[HelloStretchIdx.LIFT] = status["lift"]["pos"]
        velocities[HelloStretchIdx.LIFT] = status["lift"]["vel"]

        # Arm joints
        positions[HelloStretchIdx.ARM] = status["arm"]["pos"]
        velocities[HelloStretchIdx.ARM] = status["arm"]["vel"]

        # Wrist roll joint
        positions[HelloStretchIdx.WRIST_ROLL] = status["wrist_roll"]["pos"]
        velocities[HelloStretchIdx.WRIST_ROLL] = status["wrist_roll"]["vel"]

        # Wrist yaw joint
        positions[HelloStretchIdx.WRIST_YAW] = status["wrist_yaw"]["pos"]
        velocities[HelloStretchIdx.WRIST_YAW] = status["wrist_yaw"]["vel"]

        # Wrist pitch joint
        positions[HelloStretchIdx.WRIST_PITCH] = status["wrist_pitch"]["pos"]
        velocities[HelloStretchIdx.WRIST_PITCH] = status["wrist_pitch"]["vel"]

        # Gripper joint
        positions[HelloStretchIdx.GRIPPER] = status["gripper"]["pos"]
        velocities[HelloStretchIdx.GRIPPER] = status["gripper"]["vel"]

        # Head pan joint
        positions[HelloStretchIdx.HEAD_PAN] = status["head_pan"]["pos"]
        velocities[HelloStretchIdx.HEAD_PAN] = status["head_pan"]["vel"]

        # Head tilt joint
        positions[HelloStretchIdx.HEAD_TILT] = status["head_tilt"]["pos"]
        velocities[HelloStretchIdx.HEAD_TILT] = status["head_tilt"]["vel"]

        if self.in_manipulation_mode:
            # Get current base xyt
            xyt = self.get_base_pose()
            # Compute the relative xyt
            xyt = xyt_global_to_base(xyt, self._manip_xyt)
            # Set the base x to the x from this xyt
            positions[HelloStretchIdx.BASE_X] = xyt[0]

        return positions, velocities, efforts

    def get_base_pose(self) -> np.ndarray:
        """Base pose is the SE(2) pose of the base in world coords (x, y, theta)"""
        if self._initial_xyt is None:
            return None
        xyt = self.robot_sim.get_base_pose()
        return xyt_global_to_base(xyt, self._initial_xyt)

    def get_ee_pose(self) -> np.ndarray:
        """EE pose is the 4x4 matrix of the end effector location in world coords"""
        if self._initial_xyt is None:
            return None
        pose = self.robot_sim.get_ee_pose()
        return pose_global_to_base(pose, self._initial_xyt)

    def get_head_camera_pose(self) -> np.ndarray:
        """Get the camera pose in world coords"""
        if self._initial_xyt is None:
            return None
        pose = self.robot_sim.get_link_pose("camera_color_optical_frame")
        return pose_global_to_base(pose, self._initial_xyt)

    def get_ee_camera_pose(self) -> np.ndarray:
        """Get the end effector camera pose in world coords"""
        if self._initial_xyt is None:
            return None
        pose = self.robot_sim.get_link_pose("gripper_camera_color_optical_frame")
        return pose_global_to_base(pose, self._initial_xyt)

    def set_posture(self, posture: str) -> bool:
        """Set the posture of the robot."""

        # Assert posture in ["manipulation", "navigation"]
        if posture not in ["manipulation", "navigation"]:
            logger.error(
                f"Posture {posture} not supported. Must be in ['manipulation', 'navigation']"
            )
            return False

        # Set the posture
        if posture == "navigation":
            self.manip_to(constants.STRETCH_NAVIGATION_Q, all_joints=True)
        elif posture == "manipulation":
            self.manip_to(constants.STRETCH_PREGRASP_Q, all_joints=True)
        else:
            logger.error(f"Posture {posture} not supported")
            return False
        self.control_mode = posture
        return True

    def manip_to(self, q: np.ndarray, all_joints: bool = False, skip_gripper: bool = False) -> None:
        """Move the robot to a given joint configuration. q should be of size 11.

        Args:
            q (np.ndarray): Joint configuration to move the robot to.
        """

        # Check size

        if all_joints:
            assert len(q) == stretch_dof, f"q should be of size {stretch_dof}"
            # Move the robot to the given joint configuration
            for idx in range(3, stretch_dof):
                if idx == HelloStretchIdx.GRIPPER and skip_gripper:
                    continue
                self.robot_sim.move_to(mujoco_actuators[idx], q[idx])
        else:
            assert len(q) == len(manip_idx), f"q should be of size {len(manip_idx)}"
            # Just read the manipulator joints
            for i, idx in enumerate(manip_idx):
                if idx == HelloStretchIdx.BASE_X:
                    if not self.in_manipulation_mode:
                        logger.warning("Cannot move base by base_x alone in navigation mode")
                        continue
                    elif self._manip_xyt is None:
                        logger.error("Manipulation mode not set up correctly")
                    # Send an xyt goal: x, y, theta
                    # This is computed based on self._manip_xyt
                    xyt_delta = [q[i], 0, 0]
                    xyt_goal = xyt_base_to_global(xyt_delta, self._manip_xyt)
                    print("Manip xyt =", self._manip_xyt)
                    print("delta =", xyt_delta)
                    print("Setting base to", xyt_goal)
                    print("Current base is", self.get_base_pose())
                    self.set_goal_pose(xyt_goal, relative=False)
                else:
                    self.robot_sim.move_to(mujoco_actuators[idx], q[i])

    def __del__(self):
        self.stop()

    def stop(self):
        """Stop the server and the robot."""
        self.running = False
        self.robot_sim.stop()
        self._control_thread.join()

    @override
    def get_control_mode(self) -> str:
        """Get the control mode of the robot."""
        return self.control_mode

    @override
    def start(
        self, show_viewer_ui: bool = False, robocasa: bool = False, headless: bool = False
    ) -> None:
        self.robot_sim.start(
            show_viewer_ui=show_viewer_ui, headless=headless
        )  # This will start the simulation and open Mujoco-Viewer window
        super().start()

        # Create a thread for the control loop
        self._control_thread = threading.Thread(target=self._control_loop_thread)
        self._control_thread.start()

        self._initial_xyt = self.robot_sim.get_base_pose()

        if robocasa:
            if self._move_back_at_start:
                # When you start, move the agent back a bit
                # This is a hack!
                time.sleep(1.0)
                self.set_goal_pose(self.robocasa_start_offset, relative=True)

        while self.is_running():
            self._camera_data = self.robot_sim.pull_camera_data()
            self._status = self.robot_sim._pull_status()
            time.sleep(1 / self.simulation_rate)

    @override
    def handle_action(self, action: Dict[str, Any]):
        """Handle the action received from the client."""
        if "control_mode" in action:
            new_control_mode = action["control_mode"]
            if new_control_mode not in ["navigation", "manipulation"]:
                logger.error(f"Control mode {new_control_mode} not supported")
            # If we are switching to manipulation mode, recort base xyt
            if new_control_mode == "manipulation" and self.get_control_mode() == "navigation":
                self._manip_xyt = self.get_base_pose()
                logger.info(
                    "Switching to manipulation mode, recording initial base pose for manipulation: "
                    + str(self._manip_xyt)
                )

            self.control_mode = new_control_mode

        if "posture" in action:
            self.set_posture(action["posture"])
        if "gripper" in action:
            # Get current gripper pose
            positions, _, _ = self.get_joint_state()
            current_gripper_pos = positions[HelloStretchIdx.GRIPPER]
            target_gripper_pos = action["gripper"]
            step = 0.01
            t0 = timeit.default_timer()
            if current_gripper_pos < target_gripper_pos:
                while current_gripper_pos < target_gripper_pos:
                    current_gripper_pos += step
                    positions, _, _ = self.get_joint_state()
                    # TODO: remove debug print
                    # print(current_gripper_pos, positions[HelloStretchIdx.GRIPPER])
                    self.robot_sim.move_to("gripper", current_gripper_pos)
                    time.sleep(0.01)
                    dt = timeit.default_timer() - t0
                    if dt > 5:
                        logger.error("Gripper move took too long")
                        break
            else:
                while current_gripper_pos > target_gripper_pos:
                    current_gripper_pos -= step
                    positions, _, _ = self.get_joint_state()
                    # TODO: remove debug print
                    # print(current_gripper_pos, positions[HelloStretchIdx.GRIPPER])
                    self.robot_sim.move_to("gripper", current_gripper_pos)
                    time.sleep(0.02)
                    dt = timeit.default_timer() - t0
                    if dt > 5:
                        logger.error("Gripper move took too long")
                        break

        if "save_map" in action:
            logger.warning("Saving map not supported in Mujoco simulation")
        elif "load_map" in action:
            logger.warning("Loading map not supported in Mujoco simulation")
        elif "say" in action:
            # self.text_to_speech.say_async(action["say"])
            do_nothing = True
        if "joint" in action:
            # Move the robot to the given joint configuration
            # Only send the manipulator joints, not gripper or head
            print("[ROBOT] Moving to joint configuration", action["joint"])
            self.manip_to(action["joint"], all_joints=False)
        if "head_to" in action:
            self.robot_sim.move_to("head_pan", action["head_to"][0])
            self.robot_sim.move_to("head_tilt", action["head_to"][1])
        if "base_velocity" in action:
            self.robot_sim.set_base_velocity(
                v_linear=action["base_velocity"]["v"], omega=action["base_velocity"]["w"]
            )
        elif "xyt" in action:
            # Set the goal pose for the simulated velocity controller
            # If relative motion is set, the goal is relative to the current pose
            relative_motion = action.get("nav_relative", False)
            # We pass goals and let the control thread compute velocities
            self.set_goal_pose(action["xyt"], relative=relative_motion)

    @override
    def get_full_observation_message(self) -> Dict[str, Any]:
        """Get the full observation message for the robot. This includes the full state of the robot, including images and depth images."""
        cam_data = self._camera_data
        if cam_data is None:
            return None

        rgb = cam_data["cam_d435i_rgb"]
        depth = cam_data["cam_d435i_depth"]
        width, height = rgb.shape[:2]

        # Convert depth into int format
        depth = (depth * 1000).astype(np.uint16)

        # Get the joint state
        positions, _, _ = self.get_joint_state()

        # Make both into jpegs
        rgb = compression.to_jpg(rgb)
        depth = compression.to_jp2(depth)

        xyt = self.get_base_pose()

        # Get the other fields from an observation
        message = {
            "rgb": rgb,
            "depth": depth,
            "camera_K": cam_data["cam_d435i_K"],
            "camera_pose": self.get_head_camera_pose(),
            "ee_pose": self.get_ee_pose(),
            "joint": positions,
            "gps": xyt[:2],
            "compass": np.array([xyt[2]]),
            "rgb_width": width,
            "rgb_height": height,
            "control_mode": self.get_control_mode(),
            "last_motion_failed": False,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self.base_controller_at_goal(),
            "is_simulation": True,
            "lidar_points": None,
            "lidar_timestamp": None,
        }
        return message

    @override
    def get_state_message(self) -> Dict[str, Any]:
        """Get the state message for the robot. This is a smalll message that includes floating point information and booleans like if the robot is homed."""
        q, dq, eff = self.get_joint_state()
        message = {
            "base_pose": self.get_base_pose(),
            "ee_pose": self.get_ee_pose(),
            "joint_positions": q,
            "joint_velocities": dq,
            "joint_efforts": eff,
            "control_mode": self.get_control_mode(),
            "at_goal": self.base_controller_at_goal(),
            "is_homed": True,
            "is_runstopped": False,
            "step": self._last_step,
        }
        return message

    @override
    def get_servo_message(self) -> Dict[str, Any]:
        """Get messages for e2e policy learning and visual servoing. These are images and depth images, but lower resolution than the large full state observations, and they include the end effector camera."""

        cam_data = self._camera_data
        if cam_data is None:
            return None

        head_color_image = cam_data["cam_d435i_rgb"]
        head_depth_image = cam_data["cam_d435i_depth"]
        ee_color_image = cam_data["cam_d405_rgb"]
        ee_depth_image = cam_data["cam_d405_depth"]

        # Adapt color so we can use higher shutter speed
        # TODO: do we need this? Probably not.
        # ee_color_image = adjust_gamma(ee_color_image, 2.5)

        ee_color_image, ee_depth_image = self._rescale_color_and_depth(
            ee_color_image, ee_depth_image, self.ee_image_scaling
        )
        head_color_image, head_depth_image = self._rescale_color_and_depth(
            head_color_image, head_depth_image, self.image_scaling
        )

        # Conversion
        ee_depth_image = (ee_depth_image * 1000).astype(np.uint16)
        head_depth_image = (head_depth_image * 1000).astype(np.uint16)

        # Compress the images
        compressed_ee_depth_image = compression.to_jp2(ee_depth_image)
        compressed_ee_color_image = compression.to_jpg(ee_color_image)
        compressed_head_depth_image = compression.to_jp2(head_depth_image)
        compressed_head_color_image = compression.to_jpg(head_color_image)

        # Get position info
        positions, _, _ = self.get_joint_state()

        # Get the camera matrices
        head_rgb_K = cam_data["cam_d435i_K"]
        head_dpt_K = cam_data["cam_d435i_K"]
        ee_rgb_K = cam_data["cam_d405_K"]
        ee_dpt_K = cam_data["cam_d405_K"]

        message = {
            "ee_cam/color_camera_K": scale_camera_matrix(ee_rgb_K, self.ee_image_scaling),
            "ee_cam/depth_camera_K": scale_camera_matrix(ee_dpt_K, self.ee_image_scaling),
            "ee_cam/color_image": compressed_ee_color_image,
            "ee_cam/depth_image": compressed_ee_depth_image,
            "ee_cam/color_image/shape": ee_color_image.shape,
            "ee_cam/depth_image/shape": ee_depth_image.shape,
            "ee_cam/image_scaling": self.ee_image_scaling,
            "ee_cam/depth_scaling": self.ee_depth_scaling,
            "ee_cam/pose": self.get_ee_camera_pose(),
            "ee/pose": self.get_ee_pose(),
            "head_cam/color_camera_K": scale_camera_matrix(head_rgb_K, self.image_scaling),
            "head_cam/depth_camera_K": scale_camera_matrix(head_dpt_K, self.image_scaling),
            "head_cam/color_image": compressed_head_color_image,
            "head_cam/depth_image": compressed_head_depth_image,
            "head_cam/color_image/shape": head_color_image.shape,
            "head_cam/depth_image/shape": head_depth_image.shape,
            "head_cam/image_scaling": self.image_scaling,
            "head_cam/depth_scaling": self.depth_scaling,
            "head_cam/pose": self.get_head_camera_pose(),
            "robot/config": positions,
            "is_simulation": True,
        }
        return message

    @override
    def is_running(self) -> bool:
        """Check if the server is running. Will be used to make sure inner loops terminate.

        Returns:
            bool: True if the server is running, False otherwise."""
        return self.running and self.robot_sim.is_running()


@click.command()
@click.option("--send_port", default=4401, help="Port to send messages to clients")
@click.option("--recv_port", default=4402, help="Port to receive messages from clients")
@click.option("--send_state_port", default=4403, help="Port to send state-only messages to clients")
@click.option("--send_servo_port", default=4404, help="Port to send images for visual servoing")
@click.option("--use_remote_computer", default=True, help="Whether to use a remote computer")
@click.option("--verbose", default=False, help="Whether to print verbose messages", is_flag=True)
@click.option("--image_scaling", default=0.5, help="Scaling factor for images")
@click.option("--ee_image_scaling", default=0.5, help="Scaling factor for end-effector images")
@click.option("--depth_scaling", default=0.001, help="Scaling factor for depth images")
@click.option(
    "--scene_path", default=None, help="Provide a path to mujoco scene file with stretch.xml"
)
@click.option(
    "--use-robocasa",
    "--use_robocasa",
    default=False,
    help="Use robocasa for generating a scene",
    is_flag=True,
)
@click.option(
    "--robocasa-task",
    "--robocasa_task",
    default="PnPCounterToCab",
    help="Robocasa task to generate",
)
@click.option(
    "--robocasa-style", "--robocasa_style", type=int, default=1, help="Robocasa style to generate"
)
@click.option(
    "--robocasa-layout",
    "--robocasa_layout",
    type=int,
    default=1,
    help="Robocasa layout to generate",
)
@click.option(
    "--show-viewer-ui",
    "--show_viewier_ui",
    default=False,
    help="Show the Mujoco viewer UI",
    is_flag=True,
)
@click.option("--headless", default=False, help="Run the simulation headless", is_flag=True)
@click.option("--seed", default=0, help="Seed for the simulation")
@click.option(
    "--robocasa-write-to-xml",
    default=False,
    help="Write the generated scene to an xml file",
    is_flag=True,
)
def main(
    send_port: int,
    recv_port: int,
    send_state_port: int,
    send_servo_port: int,
    use_remote_computer: bool,
    verbose: bool,
    image_scaling: float,
    ee_image_scaling: float,
    depth_scaling: float,
    scene_path: str,
    use_robocasa: bool,
    robocasa_task: str,
    robocasa_style: int,
    robocasa_layout: int,
    robocasa_write_to_xml: bool,
    show_viewer_ui: bool,
    headless: bool = False,
    seed: int = 0,
):

    scene_model = None
    objects_info = None

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if use_robocasa:
        if not robocasa_write_to_xml and (scene_path is not None and len(scene_path) > 0):
            scene_model = load_model_from_xml(scene_path)
        else:
            scene_model, scene_xml, objects_info = model_generation_wizard(
                task=robocasa_task,
                style=robocasa_style,
                layout=robocasa_layout,
                write_to_file=scene_path,
            )

    # If no scene path
    if scene_path is None or len(scene_path) == 0:
        scene_path = sim_utils.get_default_scene_path()

    server = MujocoZmqServer(
        send_port,
        recv_port,
        send_state_port,
        send_servo_port,
        use_remote_computer=use_remote_computer,
        verbose=verbose,
        image_scaling=image_scaling,
        ee_image_scaling=ee_image_scaling,
        depth_scaling=depth_scaling,
        scene_path=scene_path,
        scene_model=scene_model,
        objects_info=objects_info,
    )
    try:
        server.start(
            show_viewer_ui=show_viewer_ui,
            robocasa=use_robocasa,
            headless=headless,
        )

    except KeyboardInterrupt:
        server.robot_sim.stop()


if __name__ == "__main__":
    main()
