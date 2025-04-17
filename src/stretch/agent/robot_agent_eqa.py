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

from stretch.agent.manipulation.dynamem_manipulation.dynamem_manipulation import (
    DynamemManipulationWrapper as ManipulationWrapper,
)
from stretch.agent.robot_agent_dynamem import RobotAgent as RobotAgentBase
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractGraspClient, AbstractRobotClient

# from stretch.llms import OpenaiClient
from stretch.mapping.instance import Instance
from stretch.mapping.voxel import SparseVoxelMapProxy
from stretch.mapping.voxel.voxel_eqa import SparseVoxelMapEQA as SparseVoxelMap
from stretch.mapping.voxel.voxel_map_eqa import SparseVoxelMapNavigationSpace
from stretch.motion.algo.a_star import AStar

# from stretch.perception.detection.owl import OwlPerception
from stretch.perception.encoders.masksiglip2_encoder import MaskSiglip2Encoder
from stretch.perception.wrapper import OvmmPerception

# Manipulation hyperparameters
INIT_LIFT_POS = 0.45
INIT_WRIST_PITCH = -1.57
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65


class RobotAgent(RobotAgentBase):
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        obs_sub_port: int = 4450,
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
        # For placing
        self.owl_sam_detector = None

        # if self.parameters.get("encoder", None) is not None:
        #     self.encoder: BaseImageTextEncoder = get_encoder(
        #         self.parameters["encoder"], self.parameters.get("encoder_args", {})
        #     )
        # else:
        #     self.encoder: BaseImageTextEncoder = None
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

        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + server_ip + ":" + str(manip_port))

        if re == 1 or re == 2:
            stretch_gripper_max = 0.3
            end_link = "link_straight_gripper"
        else:
            stretch_gripper_max = 0.64
            end_link = "link_gripper_s3_body"
        self.transform_node = end_link
        self.manip_wrapper = ManipulationWrapper(
            self.robot, stretch_gripper_max=stretch_gripper_max, end_link=end_link
        )
        self.robot.move_to_nav_posture()

        self.reset_object_plans()

        self.re = re

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._running = True

        # PROMPT = """
        #     Assume there is an agent doing Question Answering in an environment.
        #     When it receives a question, you need to tell the agent few objects (preferably 1-3) it needs to pay special attention to.
        #     Example:
        #         Input: Where is the pen?
        #         Output: pen

        #         Input: Is there grey cloth on cloth hanger?
        #         Output: gery cloth,cloth hanger
        # """
        # self.llm_client = OpenaiClient(PROMPT, model="gpt-4o-mini")

        self._start_threads()

    def create_obstacle_map(self, parameters):
        if self.manipulation_only:
            self.encoder = None
        else:
            self.encoder = MaskSiglip2Encoder(device=self.device, version="giant")
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
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="QA"),
                # rrb.Spatial2DView(name="image", origin="/observation_similar_to_text"),
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
        # time.sleep(0.3)
        obs = self.robot.get_observation()
        self.obs_count += 1
        rgb, depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
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

    def look_around(self):
        print("*" * 10, "Look around to check", "*" * 10)
        for pan in [0.6, -0.2, -1.0, -1.8]:
            tilt = -0.6
            self.robot.head_to(pan, tilt, blocking=True)
            self.update()

    def look_and_update(self):
        print("*" * 10, "Look around to check", "*" * 10)
        self.robot.head_to(0, -0.3, blocking=True)
        self.update()

    def rotate_in_place(self):
        print("*" * 10, "Rotate in place", "*" * 10)
        xyt = self.robot.get_base_pose()
        self.robot.head_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for i in range(8):
            xyt[2] += 2 * np.pi / 8
            self.robot.move_base_to(xyt, blocking=True)
            if not self._realtime_updates:
                self.update()

    def execute_action(
        self,
        text: str,
    ):
        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        self.robot.switch_to_navigation_mode()

        start = self.robot.get_base_pose()
        res = self.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            res = self.process_text("", start)

        if len(res) > 0:
            print("Plan successful!")
            if len(res) >= 2 and np.isnan(res[-2]).all():
                if len(res) > 2:
                    self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        blocking=True,
                    )

                return True, res[-1]
            else:
                # print(res)
                # res[-1][2] += np.pi / 2
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=True,
                )
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

    def process_text(self, text, start_pose):
        """
        Process the text query and return the trajectory for the robot to follow.
        """

        print("Processing", text, "starts")

        self.rerun_visualizer.clear_identity("world/object")
        self.rerun_visualizer.clear_identity("world/robot_start_pose")
        self.rerun_visualizer.clear_identity("world/direction")
        self.rerun_visualizer.clear_identity("robot_monologue")
        self.rerun_visualizer.clear_identity("/observation_similar_to_text")

        debug_text = ""
        mode = "navigation"
        obs = None
        localized_point = None
        waypoints = None

        if text is not None and text != "" and self.space.traj is not None:
            print("saved traj", self.space.traj)
            traj_target_point = self.space.traj[-1]
            if self.voxel_map.verify_point(text, traj_target_point):
                localized_point = traj_target_point
                debug_text += "## Last visual grounding results looks fine so directly use it.\n"

        print("Target verification finished")

        if text is not None and text != "" and localized_point is None:
            (
                localized_point,
                debug_text,
                obs,
                pointcloud,
            ) = self.voxel_map.localize_text(text, debug=True, return_debug=True)
            print("Target point selected!")

        # Do Frontier based exploration
        if text is None or text == "" or localized_point is None:
            debug_text += "## Navigation fails, so robot starts exploring environments.\n"
            localized_point = self.space.sample_frontier(self.planner, start_pose, text)
            mode = "exploration"

        if obs is not None and mode == "navigation":
            print(obs, len(self.voxel_map.observations))
            obs = self.voxel_map.find_obs_id_for_text(text)
            rgb = self.voxel_map.observations[obs - 1].rgb
            self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb)

        if localized_point is None:
            return []

        # TODO: Do we really need this line?
        if len(localized_point) == 2:
            localized_point = np.array([localized_point[0], localized_point[1], 0])

        point = self.space.sample_navigation(start_pose, self.planner, localized_point)

        print("Navigation endpoint selected")

        waypoints = None

        if point is None:
            res = None
            print("Unable to find any target point, some exception might happen")
        else:
            res = self.planner.plan(start_pose, point)

        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
        elif res is not None:
            waypoints = None
            print("[FAILURE]", res.reason)
        # If we are navigating to some object of interest, send (x, y, z) of
        # the object so that we can make sure the robot looks at the object after navigation
        traj = []
        if waypoints is not None:

            self.rerun_visualizer.log_custom_pointcloud(
                "world/object",
                [localized_point[0], localized_point[1], 1.5],
                torch.Tensor([1, 0, 0]),
                0.1,
            )

            finished = len(waypoints) <= 8 and mode == "navigation"
            if finished:
                self.space.traj = None
            else:
                self.space.traj = waypoints[8:] + [[np.nan, np.nan, np.nan], localized_point]
            if not finished:
                waypoints = waypoints[:8]
            traj = self.planner.clean_path_for_xy(waypoints)
            if finished:
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            print("Planned trajectory:", traj)

        # Talk about what you are doing, as the robot.
        if self.robot is not None:
            if text is not None and text != "":
                self.robot.say("I am looking for a " + text + ".")
            else:
                self.robot.say("I am exploring the environment.")

        if text is not None and text != "":
            debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
        else:
            debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
        debug_text = "# Robot's monologue: \n" + debug_text
        self.rerun_visualizer.log_text("robot_monologue", debug_text)

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

        return traj

    def run_eqa(self, question):
        rr.init("Stretch_robot", recording_id=uuid4(), spawn=True)

        self.robot.switch_to_navigation_mode()

        return self.run_eqa_vlm_planner(question)

    def run_eqa_vlm_planner(self, question, max_planning_steps: int = 5):
        answer_output = None
        for cnt_step in range(max_planning_steps):
            click.secho(
                f"Overall step {cnt_step}",
                fg="blue",
            )

            if not self._realtime_updates:
                self.robot.look_front()
                self.look_around()
                self.robot.look_front()
                self.robot.switch_to_navigation_mode()

            try:
                reasoning, answer, confidence, confidence_reasoning = self.voxel_map.query_answer(
                    question
                )
            except:
                reasoning, answer, confidence, confidence_reasoning = (
                    "Exception happens in LLM querying",
                    "Unknown",
                    False,
                    "",
                )
            answer_output = (
                "## Question: "
                + question
                + "\n## Answer: "
                + answer
                + "\n## Reasoning for the answer: "
                + reasoning
                + "\n## Reasoning for the confidence: "
                + confidence_reasoning
            )

            self.rerun_visualizer.log_text("QA", answer_output)

            if confidence:
                break

            start_pose = self.robot.get_base_pose()
            target_point = self.space.sample_frontier(
                self.planner, start_pose, text="answering the question '" + question + "'"
            )
            target_theta = self.space.sample_navigation(start_pose, self.planner, target_point)[-1]
            print(target_theta)
            movement_step = 0
            while movement_step < 5:
                start_pose = self.robot.get_base_pose()
                movement_step += 1
                self.update()
                finished = self.navigate_to_target_pose(target_point, start_pose, target_theta)
                if finished:
                    break

    def navigate_to_target_pose(
        self,
        target_pose: Optional[Union[torch.Tensor, np.ndarray, list, tuple]],
        start_pose: Optional[Union[torch.Tensor, np.ndarray, list, tuple]],
        target_theta: Optional[float] = None,
    ):
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
            if finished and target_theta is not None:
                traj[-1][2] = target_theta
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

            self.robot.execute_trajectory(
                traj,
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
                blocking=True,
            )

        return finished
