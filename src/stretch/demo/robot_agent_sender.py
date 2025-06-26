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

import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import click
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zmq
from PIL import Image

from stretch.agent.robot_agent_dynamem import RobotAgent as RobotAgentBase
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractGraspClient, AbstractRobotClient
from stretch.mapping.instance import Instance
from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from stretch.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from stretch.mapping.voxel import SparseVoxelMapProxy
from stretch.motion.algo.a_star import AStar
from stretch.perception.encoders.masksiglip_encoder import MaskSiglipEncoder
from stretch.perception.wrapper import OvmmPerception
from stretch.demo.communication_util import load_socket, send_array, recv_array, send_rgb_img, recv_rgb_img, send_depth_img, recv_depth_img, send_everything, recv_everything


class RobotAgent(RobotAgentBase):
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        save_rerun: bool = False,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        re: int = 3,
        manip_port: int = 5557,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        mllm: bool = False,
        manipulation_only: bool = False,
    ):
        self.reset_object_plans()
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.grasp_client = grasp_client
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        self.rerun_visualizer = self.robot._rerun
        self.setup_custom_blueprint()

        self.mllm = mllm
        self.manipulation_only = manipulation_only

        context = zmq.Context()
        self.img_socket = context.socket(zmq.REQ)
        self.img_socket.connect("tcp://" + str(server_ip) + ":" + str(5555))
        self.text_socket = context.socket(zmq.REQ)
        self.text_socket.connect("tcp://" + str(server_ip) + ":" + str(5556))
        self.pose_socket = context.socket(zmq.REQ)
        self.pose_socket.connect("tcp://" + str(server_ip) + ":" + str(5554))
        self.manip_socket = context.socket(zmq.REQ)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists("dynamem_log"):
            os.makedirs("dynamem_log")

        if log is None:
            current_datetime = datetime.now()
            self.log = "dynamem_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = "dynamem_log/" + log

        # ==============================================
        self.obs_count = 0
        self.obs_history: List[Observations] = []

        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.use_scene_graph = self.parameters["use_scene_graph"]
        self.tts = get_text_to_speech(self.parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory
        self._realtime_updates = realtime_updates

        # ==============================================
        # Update configuration
        # If true, the head will sweep on update, collecting more information.
        self._sweep_head_on_update = parameters["agent"]["sweep_head_on_update"]

        # ==============================================
        # Task-level parameters
        # Grasping parameters
        self.current_receptacle: Instance = None
        self.current_object: Instance = None
        self.target_object = None
        self.target_receptacle = None
        # ==============================================

        # Parameters for feature matching and exploration
        self._is_match_threshold = parameters.get("encoder_args/feature_match_threshold", 0.05)
        self._grasp_match_threshold = parameters.get(
            "encoder_args/grasp_feature_match_threshold", 0.05
        )

        # Expanding frontier - how close to frontier are we allowed to go?
        self._default_expand_frontier_size = parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = parameters["motion_planner"]["frontier"]["step_dist"]
        self._manipulation_radius = parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._voxel_size = parameters["voxel_size"]

        self.robot.move_to_nav_posture()

        self.reset_object_plans()

        self.re = re

        self.save_rerun = save_rerun

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._running = True

        self.rerun_iter = 0

        self._start_threads()

    def setup_custom_blueprint(self):
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="Semantic memory", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="QA", origin="QA"),
                rrb.Spatial2DView(name="images relevant to QA", origin="relevant_images"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="/world/head_camera"),
                rrb.Spatial2DView(name="ee_rgb", origin="/world/ee_camera"),
            ),
            column_shares=[2, 1, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=True,
        )
        rr.send_blueprint(my_blueprint)

    def update(self):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        # Sleep some time for the robot camera to focus
        obs = self.robot.get_observation()
        self.obs_count += 1
        rgb, depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
        send_everything(self.img_socket, rgb, depth, K, camera_pose)

    def look_around(self):
        print("*" * 10, "Look around to check", "*" * 10)
        for pan in [0.6, -0.2, -1.0, -1.8]:
            tilt = -0.6
            self.robot.head_to(pan, tilt, blocking=True)
            self.update()

    def rotate_in_place(self):
        print("*" * 10, "Rotate in place", "*" * 10)
        if self.save_rerun:
            if not os.path.exists(self.log):
                os.makedirs(self.log)
            rr.save(self.log + "/" + "data_" + str(self.rerun_iter) + ".rrd")
        xyt = self.robot.get_base_pose()
        self.robot.head_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for i in range(8):
            xyt[2] += 2 * np.pi / 8
            self.robot.move_base_to(xyt, blocking=True)
            if not self._realtime_updates:
                self.update()
        self.rerun_iter += 1

    def run_eqa(self, question, max_planning_steps: int = 5):
        """
        API for calling EQA module
        """

        self.robot.switch_to_navigation_mode()

        for cnt_step in range(max_planning_steps):
            click.secho(
                f"Overall step {cnt_step}",
                fg="blue",
            )
            answer, confidence = self.run_eqa_one_iter(question)
            if confidence:
                self.robot.say("The answer to " + question + " is " + answer)
                break

        return None, None

    def run_eqa_one_iter(self, question, max_movement_step: int = 5):
        answer_output = None

        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        self.text_socket.send_string(question)
        self.text_socket.recv_string()
        send_array(self.text_socket, self.robot.get_base_pose())
        answer = self.text_socket.recv_string()
        self.text_socket.send_string("")
        confidence = self.text_socket.recv_string().lower() == "true"

        if confidence:
            return answer, confidence
        else:
            self.text_socket.send_string("")
            target_point = recv_array(self.text_socket) 

        movement_step = 0
        while movement_step < max_movement_step:
            start_pose = self.robot.get_base_pose()
            movement_step += 1
            self.update()
            finished = self.navigate_to_target_pose(target_point, start_pose)
            if finished:
                break

        return answer, confidence

    def navigate_to_target_pose(
        self,
        target_pose: Optional[Union[torch.Tensor, np.ndarray, list, tuple]],
        start_pose: Optional[Union[torch.Tensor, np.ndarray, list, tuple]]
    ):
        send_array(self.pose_socket, start_pose)
        self.pose_socket.recv_string()
        send_array(self.pose_socket, target_pose)
        
        traj = recv_array(self.pose_socket)

        self.robot.execute_trajectory(
                traj,
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
                blocking=True,
        )

        self.pose_socket.send_string("")
        return self.pose_socket.recv_string().lower() == 'true'
