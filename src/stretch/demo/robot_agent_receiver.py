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
from stretch.core.robot import AbstractGraspClient
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
import threading


class RobotAgent(RobotAgentBase):
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
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
        self.grasp_client = grasp_client
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        from stretch.visualization.rerun import RerunVisualizer

        self.rerun_visualizer = RerunVisualizer()
        self.setup_custom_blueprint()

        self.mllm = mllm
        self.manipulation_only = manipulation_only

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists("dynamem_log"):
            os.makedirs("dynamem_log")

        if log is None:
            current_datetime = datetime.now()
            self.log = "dynamem_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = "dynamem_log/" + log

        self.create_obstacle_map(parameters)

        # Create voxel map information for multithreaded access
        self._voxel_map_lock = Lock()
        self.voxel_map_proxy = SparseVoxelMapProxy(self.voxel_map, self._voxel_map_lock)

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

        self.reset_object_plans()
        
        self.img_socket = load_socket(5555)
        self.text_socket = load_socket(5556)
        self.pose_socket = load_socket(5554)

        self.img_thread = threading.Thread(target=self._recv_image)
        self.img_thread.daemon = True
        self.img_thread.start()

        self.navigation_thread = threading.Thread(target=self._pose_server)
        self.navigation_thread.daemon = True
        self.navigation_thread.start()

        self.save_rerun = save_rerun

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._running = True

        self.rerun_iter = 0

        self._start_threads()

    def create_obstacle_map(self, parameters):
        if self.manipulation_only:
            self.encoder = None
        else:
            self.encoder = MaskSiglipEncoder(device=self.device, version="so400m")
        semantic_memory_resolution = 0.1
        image_shape = (360, 270)
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            semantic_memory_resolution=semantic_memory_resolution,
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            grid_resolution=0.1,
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            detection=None,
            encoder=self.encoder,
            image_shape=image_shape,
            log=self.log,
            mllm=self.mllm,
            # Important as we want to generate visual clues
            run_eqa=True,
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
        )
        self.planner = AStar(self.space)

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

    def _recv_image(self):
        while True:
            rgb, depth, intrinsics, pose = recv_everything(self.img_socket)
            print("Image received")
            self.update(rgb, depth, intrinsics, pose)

    def update(self, rgb, depth, K, camera_pose):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""

        self.obs_count += 1
        self.voxel_map.process_rgbd_images(rgb, depth, K, camera_pose)
        if self.voxel_map.voxel_pcd._points is not None:
            self.rerun_visualizer.update_voxel_map(space=self.space)
        if self.voxel_map.semantic_memory._points is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/semantic_memory/pointcloud",
                self.voxel_map.semantic_memory._points.detach().cpu(),
                self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                0.03,
            )

    def patch_images(self, images: List[Image.Image], patch_size=(480, 640), gap=5):
        """
        Patch a list of PIL Images into a numpy array, used for dicrod bot
        """
        # Resize all images to the same patch size
        images = [img.resize(patch_size) for img in images]

        # Calculate total width and height
        n_images = len(images)
        total_width = patch_size[0] * n_images + gap * (n_images - 1)
        total_height = patch_size[1]

        # Create a blank canvas
        canvas = Image.new("RGB", (total_width, total_height))

        # Paste images side-by-side
        for idx, img in enumerate(images):
            x = idx * (patch_size[0] + gap)
            canvas.paste(img, (x, 0))

        # Convert to numpy array
        return np.array(canvas)

    def recv_text(self):
        question = self.text_socket.recv_string()
        print('fuck', question, '\n\n')
        self.text_socket.send_string('Text recevied, waiting for robot pose')
        start_pose = recv_array(self.text_socket)
        answer, discord_text, relevant_images, confidence, target_point = self.query(question, start_pose)
        self.text_socket.send_string(answer)
        self.text_socket.recv_string()
        self.text_socket.send_string(str(confidence))
        if not confidence:
            self.text_socket.recv_string()
            send_array(self.text_socket, target_point)

    def query(self, question, start_pose):
        answer_output = None

        try:
            (
                reasoning,
                answer,
                confidence,
                confidence_reasoning,
                target_point,
                relevant_images,
            ) = self.voxel_map.query_answer(question, start_pose, self.planner)
        except:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images = (
                "Exception happens in LLM querying",
                "Unknown",
                False,
                "",
                self.space.sample_frontier(self.planner, start_pose, text=None),
                [],
            )

        # Log the texts to rerun visualizer
        confidence_text = (
            "I am confident with the answer" if confidence else "I am NOT confident with the answer"
        )

        reasoning_output = (
            "\n#### Reasoning for the answer: " + reasoning
            if confidence
            else "\n#### Reasoning for the confidence: " + confidence_reasoning
        )

        answer_output = (
            "#### **Question:** "
            + question
            + "\n#### **Answer:** "
            + answer
            + "\n#### **Confidence:** "
            + confidence_text
            + reasoning_output
        )

        self.rerun_visualizer.log_text("QA", answer_output)
        if len(relevant_images) != 0:
            self.rerun_visualizer.log_custom_2d_image(
                "relevant_images", self.patch_images(relevant_images)
            )

        # chat with user in the rerun
        if confidence:
            discord_text = answer + ". I believe this answer is correct because " + reasoning
        else:
            discord_text = (
                "I am not confident to answer the question because " + confidence_reasoning
            )

        discord_text += "\nI also provide relevant images here."

        if confidence:
            return answer, discord_text, relevant_images, confidence, None

        print("Target point", target_point)
        # If we want to explore non obstacles (especially frontiers), remember where we currently want to face
        obstacles, _ = self.voxel_map.get_2d_map()
        target_grid = self.voxel_map.xy_to_grid_coords((target_point[0], target_point[1]))

        return answer, discord_text, relevant_images, confidence, target_point
    
    def _pose_server(self):
        while True:
            self.navigate_to_target_pose()

    def navigate_to_target_pose(self):
        start_pose = recv_array(self.pose_socket)
        self.pose_socket.send_string("")
        target_pose = recv_array(self.pose_socket)
        res = None
        original_target_pose = target_pose
        if target_pose is not None:
            # target_pose originally represents the place where the object of interest is.
            # This line finds the pose where the robot should stop
            target_pose = self.space.sample_navigation(start_pose, self.planner, target_pose)

            # A* planning
            if target_pose is not None:
                res = self.planner.plan(start_pose, target_pose)

        # Parse A* results into traj
        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
        elif res is not None:
            waypoints = None
            print("[FAILURE]", res.reason)
        else:
            waypoints = None

        if waypoints is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/target_pose",
                [original_target_pose[0], original_target_pose[1], 1.5],
                torch.Tensor([1, 0, 0]),
                0.1,
            )

        finished = True
        if waypoints is not None:
            if not len(waypoints) <= 8:
                waypoints = waypoints[:8]
                finished = False
            traj = self.planner.clean_path_for_xy(waypoints)
            print("Planned trajectory:", traj)
        else:
            traj = None

        # draw traj on rerun and execute it
        if traj is not None:
            origins = []
            vectors = []
            for idx in range(len(traj)):
                if idx != len(traj) - 1:
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append(
                        [traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0]
                    )
            self.rerun_visualizer.log_arrow3D(
                "world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1
            )
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 1.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

        send_array(self.pose_socket, traj)

        return finished
    
from stretch.core.parameters import get_parameters
parameters = get_parameters("dynav_config.yaml")
agent = RobotAgent(parameters=parameters)
while True:
    agent.recv_text()
