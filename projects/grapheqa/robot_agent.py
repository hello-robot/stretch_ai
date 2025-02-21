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
import time
import timeit
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import click
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from planners import VLMPLannerEQAGPT
from scene_graph import SceneGraphSim

# )
from stretch.agent.robot_agent import RobotAgent as RobotAgentBase
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractRobotClient
from stretch.llms import OpenaiClient
from stretch.mapping.instance import Instance
from stretch.mapping.scene_graph import SceneGraph
from stretch.mapping.voxel import SparseVoxelMapProxy
from stretch.motion.algo.a_star import AStar

# from stretch.perception.captioners import VitGPT2Captioner
from stretch.perception.captioners import GitCaptioner
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.perception.wrapper import OvmmPerception


class RobotAgent(RobotAgentBase):
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = True,
        realtime_updates: bool = False,
        obs_sub_port: int = 4450,
        re: int = 3,
        manip_port: int = 5557,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        mllm: bool = False,
    ):
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        self.rerun_visualizer = self.robot._rerun
        self.setup_custom_blueprint()

        self.mllm = mllm
        # For placing
        # self.owl_sam_detector = none

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._realtime_updates = realtime_updates

        if log is None:
            current_datetime = datetime.now()
            self.log = "grapheqa_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = "grapheqa_log/" + log

        if not os.path.exists(self.log):
            os.makedirs(self.log)

        self.create_obstacle_map(parameters)

        # ==============================================
        self.obs_count = 0
        if realtime_updates:
            self.obs_history: List[Observations] = []

        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.tts = get_text_to_speech(self.parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory

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

        if self._realtime_updates:
            self.obs_count = 0
            self._matched_vertices_obs_count: Dict[float, int] = dict()
            self._matched_observations_poses: List[np.ndarray] = []
            self._head_not_moving_tolerance = float(
                self.parameters.get("motion/joint_thresholds/head_not_moving_tolerance", 1.0e-4)
            )
            self._realtime_matching_distance = self.parameters.get(
                "agent/realtime/matching_distance", 0.5
            )
            self._maximum_matched_observations = self.parameters.get(
                "agent/realtime/maximum_matched_observations", 50
            )
            self._realtime_temporal_threshold = self.parameters.get(
                "agent/realtime/temporal_threshold", 0.1
            )
            self._camera_pose_match_threshold = self.parameters.get(
                "agent/realtime/camera_pose_match_threshold", 0.05
            )

        self.robot.move_to_nav_posture()

        # Store the current scene graph computed from detected objects
        self.scene_graph = SceneGraph(self.parameters, [])  # type: ignore
        self.sg_sim = SceneGraphSim(
            output_path=self.log,
            scene_graph=self.scene_graph,
            robot=self.robot,
            enrich_object_labels="object",
        )
        self.vlm_planner = VLMPLannerEQAGPT(
            vlm_type="gpt-4o",
            sg_sim=self.sg_sim,
            question="",
            output_path=self.log,
        )
        PROMPT = """
            Assume there is an agent doing Question Answering in an environment.
            When it receives a question, you need to tell the agent which object it needs to pay special attention to.
            Example:
                Input: Where is the pen?
                Output: pen
        """
        self.llm_client = OpenaiClient(PROMPT, model="gpt-4o-mini")

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._running = True

        self._start_threads()

    def create_obstacle_map(self, parameters):
        self.encoder = MaskSiglipEncoder(device=self.device, version="so400m")
        self.captioner = GitCaptioner(device=self.device)

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
            encoder=self.encoder,
            captioner=self.captioner,
            log=self.log,
            mllm=self.mllm,
        )

        # Create voxel map information for multithreaded access
        self._voxel_map_lock = Lock()
        if self._realtime_updates:
            self.voxel_map = SparseVoxelMapProxy(self.voxel_map, self._voxel_map_lock)

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

    def update_map_with_pose_graph(self, verbose: bool = False):
        """Update our voxel map using a pose graph.
        Updates the map from pose graph. This is useful for when we are processing real-time updates."""

        self._obs_history_lock.acquire()
        t0 = timeit.default_timer()

        matched_obs = []

        for obs in self.obs_history:
            if obs.is_pose_graph_node:
                matched_obs.append(obs)

        self._obs_history_lock.release()

        added = 0
        for obs in matched_obs:
            if obs.is_pose_graph_node:
                if obs is not None and self.robot.in_navigation_mode():
                    self.voxel_map.add_obs(obs)
                    if self.voxel_map.voxel_pcd._points is not None:
                        self.rerun_visualizer.update_voxel_map(space=self.space)
                    added += 1
                    # If we add more than 3 images, we should probably stop
                    if added >= 3:
                        break

        robot_center = np.zeros(3)
        robot_center[:2] = self.robot.get_base_pose()[:2]

        self._update_scene_graph()
        self.scene_graph.get_relationships()

        if len(self.get_voxel_map().observations) > 0:
            self.update_rerun()

        t1 = timeit.default_timer()
        if verbose:
            print(f"Done updating scene graph. Time: {t1 - t0}")

        time.sleep(0.3)

    def update_frontiers(self):
        # torch.cuda.empty_cache()
        # gc.collect()

        grid_origin = self.voxel_map.grid_origin
        grid_resolution = self.voxel_map.grid_resolution

        frontier, outside_frontier, traversible = self.space.get_frontier()

        self.frontier_points = np.array(self.space.occupancy_map_to_3d_points(frontier))
        self.outside_frontier_points = np.array(
            self.space.occupancy_map_to_3d_points(outside_frontier)
        )
        self.traversible = np.array(self.space.occupancy_map_to_3d_points(traversible))

        # _clustered_frontiers = cluster_frontiers(self.frontier_points)
        # self.clustered_frontiers = []
        # print("Checking clustered frontiers")
        # for frontier in _clustered_frontiers:
        #     if self.space.is_valid(frontier, verbose=False):
        #         self.clustered_frontiers.append(frontier)
        # self.clustered_frontiers = np.stack(self.clustered_frontiers, axis=0)
        self.clustered_frontiers = cluster_frontiers(self.frontier_points)

    def _update_scene_graph(self):
        """Update the scene graph with the latest observations."""
        self.scene_graph.update(self.get_voxel_map().get_instances())
        # For debugging - TODO delete this code
        self.scene_graph.get_relationships(debug=False)

        for instance in self.scene_graph.instances:
            instance.name = self.semantic_sensor.get_class_name_for_id(instance.category_id)
            # print(self.semantic_sensor.get_class_name_for_id(instances_map[k].category_id))
            # print(k, instances_map[k].global_id, instances_map[k].category_id)

        # self.robot._rerun.update_scene_graph(self.scene_graph, self.semantic_sensor)

    def update(self):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        # Sleep some time for the robot camera to focus
        # time.sleep(0.3)
        obs = self.robot.get_observation()
        self.obs_count += 1
        rgb, depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
        if self.semantic_sensor is not None:
            # Semantic prediction
            obs = self.semantic_sensor.predict(obs)
        self.voxel_map.add_obs(obs)
        self._update_scene_graph()
        self.scene_graph.get_relationships()

        self.update_frontiers()
        # If there is no place to explore, set the home point as exploration point
        # TODO: Maybe set it to the place that has not been visited for a long time
        if len(self.clustered_frontiers) == 0:
            self.clustered_frontiers = np.array([[0.0, 0.0, 0.0]])
        self.sg_sim.update(frontier_nodes=self.clustered_frontiers, imgs_rgb=[obs.rgb])
        if self.voxel_map.voxel_pcd._points is not None:
            self.rerun_visualizer.update_voxel_map(space=self.space)
            self.rerun_visualizer.update_scene_graph(self.scene_graph, self.semantic_sensor)

    def look_around(self):
        print("*" * 10, "Look around to check", "*" * 10)
        # for pan in [0.6, -0.2, -1.0, -1.8]:
        pan = 0
        tilt = -0.6
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

    def run_eqa_vlm_planner(self, max_planning_steps: int = 3):
        start_pose = self.robot.get_base_pose()
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

            (
                target_pose,
                target_id,
                is_confident,
                confidence_level,
                answer,
                explanation_ans,
            ) = self.vlm_planner.get_next_action()

            answer_output = answer + "\n" + explanation_ans

            self.rerun_visualizer.log_text("QA", answer_output)

            if is_confident or (confidence_level > 0.85):
                result = f"Success"
                click.secho(
                    result,
                    fg="blue",
                )
                click.secho(
                    f"VLM Planner answer: {answer_output}",
                    fg="green",
                )
                break

            res = None
            if target_pose is not None:
                # self.robot._rerun.log_vlm_target(target_pose, format="xyz")

                # if self.robot._rerun:
                #     self.robot._rerun.log_vlm_target(target_pose, format="xyt")

                target_pose = self.space.sample_navigation(start_pose, self.planner, target_pose)

                res = self.planner.plan(start_pose, target_pose)

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
                    "world/target_pose",
                    [target_pose[0], target_pose[1], 1.5],
                    torch.Tensor([1, 0, 0]),
                    0.1,
                )

            if waypoints is not None:
                if not len(waypoints) <= 8:
                    waypoints = waypoints[:8]
                traj = self.planner.clean_path_for_xy(waypoints)
                print("Planned trajectory:", traj)
            else:
                traj = None

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

        return answer_output

    def run_eqa(self, question):
        rr.init("Stretch Robot", recording_id=uuid4(), spawn=True)

        self.robot.switch_to_navigation_mode()

        enrich_object = self.llm_client(question)

        print("Enrich Object name:", enrich_object)

        self.vlm_planner._question = question
        self.sg_sim.update_language_embedding(enrich_object)

        return self.run_eqa_vlm_planner()

    def get_voxel_map(self):
        """Return the voxel map"""
        return self.voxel_map


def cluster_frontiers(
    frontier_points, min_points_for_clustering=5, num_clusters=10, cluster_threshold=0.8
):
    # # cluster, or return none
    if len(frontier_points) < min_points_for_clustering:
        return frontier_points

    clusters = fps(frontier_points, num_clusters)

    # merge clusters if too close to each other
    clusters_new = np.empty((0, 3))
    for cluster in clusters:
        if len(clusters_new) == 0:
            clusters_new = np.vstack((clusters_new, cluster))
        else:
            clusters_array = np.array(clusters_new)
            dist = np.sqrt(np.sum((clusters_array - cluster) ** 2, axis=1))
            if np.min(dist) > cluster_threshold:
                clusters_new = np.vstack((clusters_new, cluster))
    return clusters_new


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype="int")  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float("inf")  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = ((points[last_added] - points[points_left]) ** 2).sum(
            -1
        )  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        try:
            selected = np.argmax(dists[points_left])
        except:
            import ipdb

            ipdb.set_trace()
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]
