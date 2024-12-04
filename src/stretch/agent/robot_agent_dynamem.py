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
import timeit
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zmq

from stretch.agent.robot_agent import RobotAgent as RobotAgentBase
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractGraspClient, AbstractRobotClient
from stretch.dynav.ok_robot_hw.camera import RealSenseCamera
from stretch.dynav.ok_robot_hw.global_parameters import (
    INIT_ARM_POS,
    INIT_HEAD_PAN,
    INIT_HEAD_TILT,
    INIT_LIFT_POS,
    INIT_WRIST_PITCH,
    INIT_WRIST_ROLL,
    INIT_WRIST_YAW,
    TOP_CAMERA_NODE,
)
from stretch.dynav.ok_robot_hw.robot import HelloRobot as Manipulation_Wrapper
from stretch.dynav.ok_robot_hw.utils.grasper_utils import (
    capture_and_process_image,
    move_to_point,
    pickup,
)

# from stretch.dynav.voxel_map_server import ImageProcessor as VoxelMapImageProcessor
from stretch.mapping.instance import Instance
from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from stretch.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from stretch.mapping.voxel import SparseVoxelMapProxy
from stretch.motion.algo.a_star import AStar
from stretch.perception.detection.owl import OwlPerception

# from stretch.perception.encoders import BaseImageTextEncoder, MaskSiglipEncoder, get_encoder
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.perception.wrapper import OvmmPerception


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
        self.current_state = "WAITING"

        self.rerun_visualizer = self.robot._rerun
        self.setup_custom_blueprint()

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
        self._is_match_threshold = parameters["instance_memory"]["matching"][
            "feature_match_threshold"
        ]

        # Expanding frontier - how close to frontier are we allowed to go?
        self._default_expand_frontier_size = parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = parameters["motion_planner"]["frontier"]["step_dist"]
        self._manipulation_radius = parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._voxel_size = parameters["voxel_size"]

        # self.image_processor = VoxelMapImageProcessor(
        #     rerun=True,
        #     rerun_visualizer=self.robot._rerun,
        #     log="dynamem_log/" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        #     robot=self.robot,
        # )  # type: ignore
        # self.encoder = self.image_processor.get_encoder()
        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://100.108.67.79:" + str(manip_port))

        if re == 1 or re == 2:
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

        self.reset_object_plans()

        self.re = re

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._running = True

        self._start_threads()

    def create_obstacle_map(self, parameters):
        self.encoder = MaskSiglipEncoder(device=self.device, version="so400m")
        self.detection_model = OwlPerception(device=self.device)
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
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
            detection=self.detection_model,
            encoder=self.encoder,
            log=self.log,
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )
        self.planner = AStar(self.space)

    def setup_custom_blueprint(self):
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.Spatial2DView(name="image", origin="/observation_similar_to_text"),
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

    def compute_blur_metric(self, image):
        """
        Computes a blurriness metric for an image tensor using gradient magnitudes.

        Parameters:
        - image (torch.Tensor): The input image tensor. Shape is [H, W, C].

        Returns:
        - blur_metric (float): The computed blurriness metric.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients using the Sobel operator
        Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        G = cv2.magnitude(Gx, Gy)

        # Compute the mean of gradient magnitudes
        blur_metric = G.mean()

        return blur_metric

    def update_map_with_pose_graph(self):
        """Update our voxel map using a pose graph"""

        t0 = timeit.default_timer()
        self.pose_graph = self.robot.get_pose_graph()

        t1 = timeit.default_timer()

        # Update past observations based on our new pose graph
        # print("Updating past observations")
        self._obs_history_lock.acquire()
        for idx in range(len(self.obs_history)):
            lidar_timestamp = self.obs_history[idx].lidar_timestamp
            gps_past = self.obs_history[idx].gps

            for vertex in self.pose_graph:
                if abs(vertex[0] - lidar_timestamp) < 0.05:
                    # print(f"Exact match found! {vertex[0]} and obs {idx}: {lidar_timestamp}")

                    self.obs_history[idx].is_pose_graph_node = True
                    self.obs_history[idx].gps = np.array([vertex[1], vertex[2]])
                    self.obs_history[idx].compass = np.array(
                        [
                            vertex[3],
                        ]
                    )

                    # print(
                    #     f"obs gps: {self.obs_history[idx].gps}, compass: {self.obs_history[idx].compass}"
                    # )

                    if (
                        self.obs_history[idx].task_observations is None
                        and self.semantic_sensor is not None
                    ):
                        self.obs_history[idx] = self.semantic_sensor.predict(self.obs_history[idx])
                # check if the gps is close to the gps of the pose graph node
                elif (
                    np.linalg.norm(gps_past - np.array([vertex[1], vertex[2]])) < 0.3
                    and self.obs_history[idx].pose_graph_timestamp is None
                ):
                    # print(f"Close match found! {vertex[0]} and obs {idx}: {lidar_timestamp}")

                    self.obs_history[idx].is_pose_graph_node = True
                    self.obs_history[idx].pose_graph_timestamp = vertex[0]
                    self.obs_history[idx].initial_pose_graph_gps = np.array([vertex[1], vertex[2]])
                    self.obs_history[idx].initial_pose_graph_compass = np.array(
                        [
                            vertex[3],
                        ]
                    )

                    if (
                        self.obs_history[idx].task_observations is None
                        and self.semantic_sensor is not None
                    ):
                        self.obs_history[idx] = self.semantic_sensor.predict(self.obs_history[idx])

                elif self.obs_history[idx].pose_graph_timestamp == vertex[0]:
                    # Calculate delta between old (initial pose graph) vertex gps and new vertex gps
                    delta_gps = vertex[1:3] - self.obs_history[idx].initial_pose_graph_gps
                    delta_compass = vertex[3] - self.obs_history[idx].initial_pose_graph_compass

                    # Calculate new gps and compass
                    new_gps = self.obs_history[idx].gps + delta_gps
                    new_compass = self.obs_history[idx].compass + delta_compass

                    # print(f"Updating obs {idx} with new gps: {new_gps}, compass: {new_compass}")
                    self.obs_history[idx].gps = new_gps
                    self.obs_history[idx].compass = new_compass

        t2 = timeit.default_timer()
        # print(f"Done updating past observations. Time: {t2- t1}")

        # print("Updating voxel map")
        t3 = timeit.default_timer()
        # self.voxel_map.reset()
        # for obs in self.obs_history:
        #     if obs.is_pose_graph_node:
        #         self.voxel_map.add_obs(obs)
        if len(self.obs_history) > 0:
            obs_history = self.obs_history[-5:]
            blurness = [self.compute_blur_metric(obs.rgb) for obs in obs_history]
            obs = obs_history[blurness.index(max(blurness))]
            # obs = self.obs_history[-1]
        else:
            obs = None

        self._obs_history_lock.release()

        if obs is not None and self.robot.in_navigation_mode():
            self.voxel_map.process_rgbd_images(obs.rgb, obs.depth, obs.camera_K, obs.camera_pose)

        robot_center = np.zeros(3)
        robot_center[:2] = self.robot.get_base_pose()[:2]

        t4 = timeit.default_timer()
        # print(f"Done updating voxel map. Time: {t4 - t3}")

        if self.use_scene_graph:
            self._update_scene_graph()
            self.scene_graph.get_relationships()

        t5 = timeit.default_timer()
        # print(f"Done updating scene graph. Time: {t5 - t4}")

        self._obs_history_lock.acquire()

        # print(f"Total observation count: {len(self.obs_history)}")

        # Clear out observations that are too old and are not pose graph nodes
        if len(self.obs_history) > 500:
            # print("Clearing out old observations")
            # Remove 10 oldest observations that are not pose graph nodes
            del_count = 0
            del_idx = 0
            while del_count < 15 and len(self.obs_history) > 0 and del_idx < len(self.obs_history):
                # print(f"Checking obs {self.obs_history[del_idx].lidar_timestamp}. del_count: {del_count}, len: {len(self.obs_history)}, is_pose_graph_node: {self.obs_history[del_idx].is_pose_graph_node}")
                if not self.obs_history[del_idx].is_pose_graph_node:
                    # print(f"Deleting obs {self.obs_history[del_idx].lidar_timestamp}")
                    del self.obs_history[del_idx]
                    del_count += 1
                else:
                    del_idx += 1

                    if del_idx >= len(self.obs_history):
                        break

        t6 = timeit.default_timer()
        # print(f"Done clearing out old observations. Time: {t6 - t5}")
        self._obs_history_lock.release()

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
            # for pan in [0.4, -0.4, -1.2]:
            for tilt in [-0.65]:
                self.robot.head_to(pan, tilt, blocking=True)
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
            ) = self.voxel_map.localize_A(text, debug=True, return_debug=True)
            print("Target point selected!")

        # Do Frontier based exploration
        if text is None or text == "" or localized_point is None:
            debug_text += "## Navigation fails, so robot starts exploring environments.\n"
            localized_point = self.space.sample_frontier(self.planner, start_pose, text)
            mode = "exploration"

        if obs is not None and mode == "navigation":
            print(obs, len(self.voxel_map.observations))
            obs = self.voxel_map.find_obs_id_for_A(text)
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

    def navigate(self, text, max_step=10):
        rr.init("Stretch_robot", recording_id=uuid4(), spawn=True)
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
            socket=self.manip_socket,
            hello_robot=self.manip_wrapper,
        )
        print("Place: ", rotation, translation)

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

    def get_voxel_map(self):
        """Return the voxel map"""
        return self.voxel_map

    def manipulate(
        self,
        text,
        init_tilt=INIT_HEAD_TILT,
        base_node=TOP_CAMERA_NODE,
        skip_confirmation: bool = False,
    ):
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
            socket=self.manip_socket,
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

        if skip_confirmation or input("Do you want to do this manipulation? Y or N ") != "N":
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
