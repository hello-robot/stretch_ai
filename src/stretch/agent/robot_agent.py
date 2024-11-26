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
import random
import time
import timeit
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import stretch.utils.memory as memory
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters, get_parameters
from stretch.core.robot import AbstractRobotClient
from stretch.mapping.instance import Instance
from stretch.mapping.scene_graph import SceneGraph
from stretch.mapping.voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace, SparseVoxelMapProxy
from stretch.motion import ConfigurationSpace, Planner, PlanResult
from stretch.motion.algo import Shortcut, SimplifyXYT, get_planner
from stretch.motion.kinematics import HelloStretchIdx
from stretch.perception.encoders import BaseImageTextEncoder, get_encoder
from stretch.perception.wrapper import OvmmPerception
from stretch.utils.geometry import angle_difference, xyt_base_to_global
from stretch.utils.logger import Logger
from stretch.utils.obj_centric import ObjectCentricObservations, ObjectImage
from stretch.utils.point_cloud import ransac_transform

logger = Logger(__name__)


class RobotAgent:
    """Basic demo code. Collects everything that we need to make this work."""

    _retry_on_fail: bool = False
    debug_update_timing: bool = False
    update_rerun_every_time: bool = True
    normalize_embeddings: bool = False

    # Default configuration file
    default_config_path = "default_planner.yaml"

    # Sleep times
    # This sleep is before starting a head sweep
    _before_head_motion_sleep_t = 0.25
    # much longer sleeps - do they fix things?
    _after_head_motion_sleep_t = 0.1

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Optional[Union[Parameters, Dict[str, Any]]] = None,
        semantic_sensor: Optional[OvmmPerception] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
        enable_realtime_updates: bool = False,
        create_semantic_sensor: bool = False,
        obs_sub_port: int = 4450,
    ):
        self.reset_object_plans()
        if parameters is None:
            self.parameters = get_parameters(self.default_config_path)
        elif isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.show_instances_detected = show_instances_detected

        # Create a semantic sensor
        self.semantic_sensor = semantic_sensor

        # If the semantic sensor is not provided, we will create it from the config file
        if create_semantic_sensor:
            if self.semantic_sensor is not None:
                logger.warn(
                    "Semantic sensor is already provided, but create_semantic_sensor is set to True. Ignoring the provided semantic sensor."
                )
            # Create a semantic sensor
            self.semantic_sensor = self.create_semantic_sensor()

        self.pos_err_threshold = self.parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = self.parameters["trajectory_rot_err_threshold"]
        self.current_state = "WAITING"
        self.encoder = get_encoder(
            self.parameters["encoder"], self.parameters.get("encoder_args", {})
        )
        self.obs_count = 0
        self.obs_history: List[Observations] = []
        self._matched_vertices_obs_count: Dict[float, int] = dict()
        self._matched_observations_poses: List[np.ndarray] = []
        self._maximum_matched_observations = self.parameters.get(
            "agent/realtime/maximum_matched_observations", 50
        )
        self._camera_pose_match_threshold = self.parameters.get(
            "agent/realtime/camera_pose_match_threshold", 0.05
        )
        self._head_not_moving_tolerance = float(
            self.parameters.get("motion/joint_thresholds/head_not_moving_tolerance", 1.0e-4)
        )

        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.use_scene_graph = self.parameters["use_scene_graph"]
        self.tts = get_text_to_speech(self.parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory
        self._realtime_updates = enable_realtime_updates or self.parameters.get(
            "agent/use_realtime_updates", False
        )
        if not enable_realtime_updates and self._realtime_updates:
            logger.warn("Real-time updates are enabled via command line.")
        if self._realtime_updates:
            logger.alert("Real-time updates are enabled!")

        # ==============================================
        # Update configuration
        # If true, the head will sweep on update, collecting more information.
        self._sweep_head_on_update = self.parameters.get("agent/sweep_head_on_update", False)

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

        # Expanding frontier - how close to frontier are we allowed to go?
        self._default_expand_frontier_size = self.parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = self.parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = self.parameters["motion_planner"]["frontier"]["step_dist"]
        self._manipulation_radius = self.parameters["motion_planner"]["goals"][
            "manipulation_radius"
        ]
        self._voxel_size = self.parameters.get("voxel_size", 0.05)
        self._realtime_matching_distance = self.parameters.get(
            "agent/realtime/matching_distance", 0.5
        )
        self._realtime_temporal_threshold = self.parameters.get(
            "agent/realtime/temporal_threshold", 0.1
        )

        if voxel_map is not None:
            self.voxel_map = voxel_map
        else:
            self.voxel_map = self._create_voxel_map(self.parameters)

        self._voxel_map_lock = Lock()
        self.voxel_map_proxy = SparseVoxelMapProxy(self.voxel_map, self._voxel_map_lock)

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.get_voxel_map(),
            self.robot.get_robot_model(),
            step_size=self.parameters["motion_planner"]["step_size"],
            rotation_step_size=self.parameters["motion_planner"]["rotation_step_size"],
            dilate_frontier_size=self.parameters["motion_planner"]["frontier"][
                "dilate_frontier_size"
            ],
            dilate_obstacle_size=self.parameters["motion_planner"]["frontier"][
                "dilate_obstacle_size"
            ],
            grid=self.get_voxel_map().grid,
        )

        self.reset_object_plans()

        # Is this still running?
        self._running = True

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        # Create a simple motion planner
        self.planner: Planner = get_planner(
            self.parameters.get("motion_planner/algorithm", "rrt_connect"),
            self.space,
            self.space.is_valid,
        )
        # Add shotcutting to the planner
        if self.parameters.get("motion_planner/shortcut_plans", False):
            self.planner = Shortcut(
                self.planner, self.parameters.get("motion_planner/shortcut_iter", 100)
            )
        # Add motion planner simplification
        if self.parameters.get("motion_planner/simplify_plans", False):
            self.planner = SimplifyXYT(
                self.planner,
                min_step=self.parameters["motion_planner"]["simplify"]["min_step"],
                max_step=self.parameters["motion_planner"]["simplify"]["max_step"],
                num_steps=self.parameters["motion_planner"]["simplify"]["num_steps"],
                min_angle=self.parameters["motion_planner"]["simplify"]["min_angle"],
            )

        self._start_threads()

    def _start_threads(self):
        """Create threads and locks for real-time updates."""

        # Track if we are still running
        self._running = True

        # Locks
        self._robot_lock = Lock()
        self._obs_history_lock = Lock()
        self._map_lock = Lock()

        if self._realtime_updates:
            logger.alert("Using real-time updates!")

            # Map updates
            self._update_map_thread = Thread(target=self.update_map_loop)
            self._update_map_thread.start()

            # Prune old observations
            self._prune_old_observations_thread = Thread(target=self.prune_old_observations_loop)
            self._prune_old_observations_thread.start()

            # Match pose graph
            self._match_pose_graph_thread = Thread(target=self.match_pose_graph_loop)
            self._match_pose_graph_thread.start()

            # Get observations thread
            self._get_observations_thread = Thread(target=self.get_observations_loop)
            self._get_observations_thread.start()
        else:
            self._update_map_thread = None
            self._get_observations_thread = None

    def get_robot(self) -> AbstractRobotClient:
        """Return the robot in use by this model"""
        return self.robot

    def get_voxel_map(self) -> SparseVoxelMapProxy:
        """Return the voxel map in use by this model"""
        return self.voxel_map_proxy

    def __del__(self):
        """Destructor. Clean up threads."""
        self._running = False
        # if self._update_map_thread is not None and self._update_map_thread.is_alive():
        #    self._update_map_thread.join()

    def _create_voxel_map(self, parameters: Parameters) -> SparseVoxelMap:
        """Create a voxel map from parameters.

        Args:
            parameters(Parameters): the parameters for the voxel map

        Returns:
            SparseVoxelMap: the voxel map
        """
        return SparseVoxelMap.from_parameters(
            parameters,
            self.encoder,
            voxel_size=self._voxel_size,
            use_instance_memory=self._use_instance_memory or self.semantic_sensor is not None,
        )

    @property
    def feature_match_threshold(self) -> float:
        """Return the feature match threshold"""
        return self._is_match_threshold

    @property
    def voxel_size(self) -> float:
        """Return the voxel size in meters"""
        return self._voxel_size

    def reset_object_plans(self):
        """Clear stored object planning information."""

        # Dictionary storing attempts to visit each object
        self._object_attempts: Dict[int, int] = {}
        self._cached_plans: Dict[int, PlanResult] = {}

        # Objects that cannot be reached
        self.unreachable_instances = set()

    def set_instance_as_unreachable(self, instance: Union[int, Instance]) -> None:
        """Mark an instance as unreachable.

        Args:
            instance(Union[int, Instance]): the instance to mark as unreachable
        """
        if isinstance(instance, Instance):
            instance_id = instance.id
        elif isinstance(instance, int):
            instance_id = instance
        else:
            raise ValueError("Instance must be an Instance object or an int")
        self.unreachable_instances.add(instance_id)

    def is_instance_unreachable(self, instance: Union[int, Instance]) -> bool:
        """Check if an instance is unreachable.

        Args:
            instance(Union[int, Instance]): the instance to check
        """
        if isinstance(instance, Instance):
            instance_id = instance.id
        elif isinstance(instance, int):
            instance_id = instance
        else:
            raise ValueError("Instance must be an Instance object or an int")
        return instance_id in self.unreachable_instances

    def get_encoder(self) -> BaseImageTextEncoder:
        """Return the encoder in use by this model"""
        return self.encoder

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using the encoder"""
        return self.encoder.encode_text(text)

    def encode_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encode image using the encoder"""
        return self.encoder.encode_image(image)

    def compare_features(self, feature1: torch.Tensor, feature2: torch.Tensor) -> float:
        """Compare two feature vectors using the encoder"""
        return self.encoder.compute_score(feature1, feature2)

    def get_instance_from_text(
        self,
        text_query: str,
        aggregation_method: str = "mean",
        normalize: bool = False,
        verbose: bool = True,
    ) -> Optional[tuple]:
        """Get the instance that best matches the text query.

        Args:
            text_query(str): the text query
            aggregation_method(str): how to aggregate the embeddings. Should be one of (max, mean).
            normalize(bool): whether to normalize the embeddings
            verbose(bool): whether to print debug info

        Returns:
            activation(float): the cosine similarity between the text query and the instance embedding
            instance(Instance): the instance that best matches the text query
        """
        if self.semantic_sensor is None:
            return None
        assert aggregation_method in [
            "max",
            "mean",
        ], f"Invalid aggregation method {aggregation_method}"
        encoded_text = self.encode_text(text_query).to(self.get_voxel_map().map_2d_device)
        best_instance = None
        best_activation = -1.0
        print("--- Searching for instance ---")
        for instance in self.get_voxel_map().get_instances():
            ins = instance.get_instance_id()
            emb = instance.get_image_embedding(
                aggregation_method=aggregation_method, normalize=normalize
            )
            activation = self.encoder.compute_score(emb, encoded_text)
            if activation.item() > best_activation:
                best_activation = activation.item()
                best_instance = instance
                if verbose:
                    print(
                        f" - Instance {ins} has activation {activation.item()} > {best_activation}"
                    )
            elif verbose:
                print(f" - Instance {ins} has activation {activation.item()}")
        return best_activation, best_instance

    def get_instances(self) -> List[Instance]:
        """Return all instances in the voxel map."""
        return self.get_voxel_map().get_instances()

    def get_instances_from_text(
        self,
        text_query: str,
        aggregation_method: str = "mean",
        normalize: bool = False,
        verbose: bool = True,
        threshold: float = 0.05,
    ) -> Optional[Tuple[List[float], List[Instance]]]:
        """Get all instances that match the text query.

        Args:
            text_query(str): the text query
            aggregation_method(str): how to aggregate the embeddings. Should be one of (max, mean).
            normalize(bool): whether to normalize the embeddings
            verbose(bool): whether to print debug info
            threshold(float): the minimum cosine similarity between the text query and the instance embedding

        Returns:
            matches: a list of tuples with two members:
                activation(float): the cosine similarity between the text query and the instance embedding
                instance(Instance): the instance that best matches the text query
        """
        if self.semantic_sensor is None:
            return None
        assert aggregation_method in [
            "max",
            "mean",
        ], f"Invalid aggregation method {aggregation_method}"
        activations = []
        matches = []
        # Encode the text query and move it to the same device as the instance embeddings
        encoded_text = self.encode_text(text_query).to(self.get_voxel_map().map_2d_device)
        # Compute the cosine similarity between the text query and each instance embedding
        for instance in self.get_voxel_map().get_instances():
            ins = instance.get_instance_id()
            emb = instance.get_image_embedding(
                aggregation_method=aggregation_method, normalize=normalize
            )

            # TODO: this is hacky - should probably just not support other encoders this way
            # if hasattr(self.encoder, "classify"):
            #    prob = self.encoder.classify(instance.get_best_view().get_image(), text_query)
            #    activation = prob
            # else:
            activation = self.encoder.compute_score(emb, encoded_text)

            # Add the instance to the list of matches if the cosine similarity is above the threshold
            if activation.item() > threshold:
                activations.append(activation.item())
                matches.append(instance)
                if verbose:
                    print(f" - Instance {ins} has activation {activation.item()} > {threshold}")
            elif verbose:
                print(
                    f" - Skipped instance {ins} with activation {activation.item()} < {threshold}"
                )
        return activations, matches

    def get_navigation_space(self) -> ConfigurationSpace:
        """Returns reference to the navigation space."""
        return self.space

    @property
    def navigation_space(self) -> ConfigurationSpace:
        """Returns reference to the navigation space."""
        return self.space

    def place_object(self, object_goal: Optional[str] = None, **kwargs) -> bool:
        """Try to place an object."""
        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()
        if self.grasp_client is None:
            logger.warn("Tried to place without providing a grasp client.")
            return False
        return self.grasp_client.try_placing(object_goal=object_goal, **kwargs)

    def grasp_object(self, object_goal: Optional[str] = None, **kwargs) -> bool:
        """Try to grasp a potentially specified object."""
        # Put the robot in manipulation mode
        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()
        if self.grasp_client is None:
            logger.warn("Tried to grasp without providing a grasp client.")
            return False
        return self.grasp_client.try_grasping(object_goal=object_goal, **kwargs)

    def rotate_in_place(
        self,
        steps: Optional[int] = -1,
        visualize: bool = False,
        verbose: bool = False,
        full_sweep: bool = True,
        audio_feedback: bool = False,
    ) -> bool:
        """Simple helper function to make the robot rotate in place. Do a 360 degree turn to get some observations (this helps debug the robot and create a nice map).

        Args:
            steps(int): number of steps to rotate (each step is 360 degrees / steps). If steps <= 0, use the default number of steps from the parameters.
            visualize(bool): show the map as we rotate. Default is False.

        Returns:
            executed(bool): false if we did not actually do any rotations"""
        logger.info("Rotate in place")
        if audio_feedback:
            self.robot.say_sync("Rotating in place")
        if steps is None or steps <= 0:
            # Read the number of steps from the parameters
            if self._realtime_updates:
                steps = self.parameters["agent"]["realtime_rotation_steps"]
                logger.info(f"Using real-time rotation steps: {steps}")
            else:
                steps = self.parameters["agent"]["in_place_rotation_steps"]

        step_size = 2 * np.pi / steps
        i = 0
        x, y, theta = self.robot.get_base_pose()
        if verbose:
            print(f"==== ROTATE IN PLACE at {x}, {y} ====")

        if full_sweep:
            steps += 1
        while i < steps:
            t0 = timeit.default_timer()
            self.robot.move_base_to(
                [x, y, theta + (i * step_size)],
                relative=False,
                blocking=True,
                verbose=verbose,
            )
            t1 = timeit.default_timer()

            if self.robot.last_motion_failed():
                # We have a problem!
                raise RuntimeError("Robot is stuck!")
            else:
                i += 1

            # Add an observation after the move
            if verbose:
                print(" - Navigation took", t1 - t0, "seconds")
                print(f"---- UPDATE {i+1} at {x}, {y}----")
            t0 = timeit.default_timer()
            if not self._realtime_updates:
                self.update()
            t1 = timeit.default_timer()
            if verbose:
                print("Update took", t1 - t0, "seconds")

            if visualize:
                self.get_voxel_map().show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                    instances=self.semantic_sensor is not None,
                )

        return True

    def get_command(self):
        """Get a command from config file or from user input if not specified."""
        task = self.parameters.get("task").get("command")
        if task is not None and len(task) > 0:
            return task
        else:
            return self.ask("Please type any task you want the robot to do: ")

    def show_map(self) -> None:
        """Helper function to visualize the 3d map as it stands right now."""
        self.get_voxel_map().show(
            orig=np.zeros(3),
            xyt=self.robot.get_base_pose(),
            footprint=self.robot.get_robot_model().get_footprint(),
            instances=self.semantic_sensor is not None,
        )

    def is_running(self) -> bool:
        """Return whether the robot agent is still running."""
        return self._running and self.robot.running

    def get_observations_loop(self, verbose=False) -> None:
        """Threaded function that gets observations in real-time. This is useful for when we are processing real-time updates."""
        while self.robot.running and self._running:
            obs = None
            t0 = timeit.default_timer()

            self._obs_history_lock.acquire()

            while obs is None:
                obs = self.robot.get_observation()
                # obs = self.sub_socket.recv_pyobj()
                # print(obs)
                if obs is None:
                    continue

                # obs = Observations.from_dict(obs)

                if (len(self.obs_history) > 0) and (
                    obs.lidar_timestamp == self.obs_history[-1].lidar_timestamp
                ):
                    obs = None
                t1 = timeit.default_timer()
                if t1 - t0 > 10:
                    break

                # Add a delay to make sure we don't get too many observations
                time.sleep(0.05)

            # t1 = timeit.default_timer()
            if obs is not None:
                self.obs_history.append(obs)
                self.obs_count += 1
            self._obs_history_lock.release()

            if verbose:
                print("Done getting an observation")
            time.sleep(0.05)

    def stop_realtime_updates(self):
        """Stop the update threads."""
        self._running = False

    def should_drop_observation(self, obs: Observations, pose_graph_timestamp: int) -> bool:
        """Check if we should drop an observation."""
        if pose_graph_timestamp in self._matched_vertices_obs_count:
            if (
                self._matched_vertices_obs_count[pose_graph_timestamp]
                > self._maximum_matched_observations
            ):
                return True

        # Check if there are any observations with camera poses that are too close
        if len(self._matched_observations_poses) > 0:
            for pose in self._matched_observations_poses:
                if np.linalg.norm(pose - obs.camera_pose) < self._camera_pose_match_threshold:
                    return True

        if obs.joint_velocities is not None:
            head_speed = np.linalg.norm(
                obs.joint_velocities[HelloStretchIdx.HEAD_PAN : HelloStretchIdx.HEAD_TILT]
            )

            if head_speed > self._head_not_moving_tolerance:
                return True

        return False

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

        with self._voxel_map_lock:
            self.voxel_map.reset()
            added = 0
            for obs in matched_obs:
                if obs.is_pose_graph_node:
                    self.voxel_map.add_obs(obs)
                    added += 1

        robot_center = np.zeros(3)
        robot_center[:2] = self.robot.get_base_pose()[:2]

        if self.use_scene_graph:
            self._update_scene_graph()
            self.scene_graph.get_relationships()

        if len(self.get_voxel_map().observations) > 0:
            self.update_rerun()

        t1 = timeit.default_timer()
        if verbose:
            print(f"Done updating scene graph. Time: {t1 - t0}")

    def prune_old_observations_loop(self, verbose: bool = False):
        while True:
            t0 = timeit.default_timer()
            self._obs_history_lock.acquire()

            if len(self.obs_history) > 1000:
                # Remove 50 oldest observations that are not pose graph nodes
                del_count = 0
                del_idx = 0
                while (
                    del_count < 50 and len(self.obs_history) > 0 and del_idx < len(self.obs_history)
                ):
                    # If obs is too recent, skip it
                    if self.obs_history[del_idx].lidar_timestamp - time.time() < 10:
                        del_idx += 1

                        if del_idx >= len(self.obs_history):
                            break

                        continue

                    if not self.obs_history[del_idx].is_pose_graph_node:
                        # Remove corresponding matched observation
                        if (
                            self.obs_history[del_idx].lidar_timestamp
                            in self._matched_vertices_obs_count
                        ):
                            self._matched_vertices_obs_count[
                                self.obs_history[del_idx].lidar_timestamp
                            ] -= 1

                        # Remove caemra pose from matched observations
                        for idx in range(len(self._matched_observations_poses)):
                            if (
                                np.linalg.norm(
                                    self._matched_observations_poses[idx]
                                    - self.obs_history[del_idx].camera_pose
                                )
                                < 0.1
                            ):
                                self._matched_observations_poses.pop(idx)
                                break

                        del self.obs_history[del_idx]
                        del_count += 1
                    else:
                        del_idx += 1

                        if del_idx >= len(self.obs_history):
                            break

            t1 = timeit.default_timer()
            self._obs_history_lock.release()

            if verbose:
                print("Done pruning old observations. Time: ", t1 - t0)
            time.sleep(1.0)

    def match_pose_graph_loop(self, verbose: bool = False):
        while True:
            t0 = timeit.default_timer()
            self.pose_graph = self.robot.get_pose_graph()

            # Update past observations based on our new pose graph
            self._obs_history_lock.acquire()

            for idx in range(len(self.obs_history)):
                lidar_timestamp = self.obs_history[idx].lidar_timestamp
                gps_past = self.obs_history[idx].gps

                for vertex in self.pose_graph:

                    if self.should_drop_observation(self.obs_history[idx], vertex[0]):
                        break

                    if abs(vertex[0] - lidar_timestamp) < self._realtime_temporal_threshold:
                        self.obs_history[idx].is_pose_graph_node = True
                        self.obs_history[idx].gps = np.array([vertex[1], vertex[2]])
                        self.obs_history[idx].compass = np.array(
                            [
                                vertex[3],
                            ]
                        )

                        if (
                            self.obs_history[idx].task_observations is None
                            and self.semantic_sensor is not None
                        ):
                            self.obs_history[idx] = self.semantic_sensor.predict(
                                self.obs_history[idx]
                            )

                        if vertex[0] not in self._matched_vertices_obs_count:
                            self._matched_vertices_obs_count[vertex[0]] = 0

                        self._matched_observations_poses.append(self.obs_history[idx].camera_pose)
                        self._matched_vertices_obs_count[vertex[0]] += 1
                    # check if the gps is close to the gps of the pose graph node
                    elif (
                        np.linalg.norm(gps_past - np.array([vertex[1], vertex[2]]))
                        < self._realtime_matching_distance
                        and self.obs_history[idx].pose_graph_timestamp is None
                    ):
                        self.obs_history[idx].is_pose_graph_node = True
                        self.obs_history[idx].pose_graph_timestamp = vertex[0]
                        self.obs_history[idx].initial_pose_graph_gps = np.array(
                            [vertex[1], vertex[2]]
                        )
                        self.obs_history[idx].initial_pose_graph_compass = np.array(
                            [
                                vertex[3],
                            ]
                        )

                        if (
                            self.obs_history[idx].task_observations is None
                            and self.semantic_sensor is not None
                        ):
                            self.obs_history[idx] = self.semantic_sensor.predict(
                                self.obs_history[idx]
                            )

                        if vertex[0] not in self._matched_vertices_obs_count:
                            self._matched_vertices_obs_count[vertex[0]] = 0

                        self._matched_observations_poses.append(self.obs_history[idx].camera_pose)
                        self._matched_vertices_obs_count[vertex[0]] += 1

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

            t1 = timeit.default_timer()

            self._obs_history_lock.release()

            if verbose:
                print(f"[Match] Released obs history lock. Time: {t1 - t0}.")
            time.sleep(0.05)

    def update_map_loop(self):
        """Threaded function that updates our voxel map in real-time."""
        while self.robot.running and self._running:
            with self._robot_lock:
                self.update_map_with_pose_graph()
            time.sleep(0.75)

    def update(
        self,
        visualize_map: bool = False,
        debug_instances: bool = False,
        move_head: Optional[bool] = None,
        tilt: float = -1 * np.pi / 4,
    ):
        if self._realtime_updates:
            return
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = None
        t0 = timeit.default_timer()

        steps = 0
        move_head = (move_head is None and self._sweep_head_on_update) or move_head is True
        if move_head:
            self.robot.move_to_nav_posture()
            # Pause a bit first to make sure the robot is in the right posture
            time.sleep(self._before_head_motion_sleep_t)
            # num_steps = 5
            num_steps = 11
        else:
            num_steps = 1

        while obs is None:
            obs = self.robot.get_observation()
            t1 = timeit.default_timer()
            if t1 - t0 > 10:
                logger.error("Failed to get observation")
                break

        for i in range(num_steps):
            if move_head:
                # pan = -1 * i * np.pi / 4
                pan = (np.pi / 3) + (-1 * i * np.pi / 6)
                print(f"[UPDATE] Head sweep {i} at {pan}, {tilt}")
                self.robot.head_to(pan, tilt, blocking=True)
                time.sleep(self._after_head_motion_sleep_t)
                obs = self.robot.get_observation()

            t1 = timeit.default_timer()

            self.obs_history.append(obs)
            self.obs_count += 1

            # Optionally do this
            if self.semantic_sensor is not None:
                # Semantic prediction
                obs = self.semantic_sensor.predict(obs)

            t2 = timeit.default_timer()
            self.get_voxel_map().add_obs(obs)
            t3 = timeit.default_timer()

            if not move_head:
                break

        if move_head:
            self.robot.head_to(0, tilt, blocking=True)
            x, y, theta = self.robot.get_base_pose()
            self.get_voxel_map().delete_obstacles(
                radius=0.3, point=np.array([x, y]), min_height=self.get_voxel_map().obs_min_height
            )

        if self.use_scene_graph:
            self._update_scene_graph()
            self.scene_graph.get_relationships()

        t4 = timeit.default_timer()

        # Add observation - helper function will unpack it
        if visualize_map:
            # Now draw 2d maps to show what was happening
            self.get_voxel_map().get_2d_map(debug=True)

        t5 = timeit.default_timer()

        if debug_instances:
            # We need to load and configure matplotlib here
            # to make sure that it's set up properly.
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            instances = self.get_voxel_map().get_instances()
            for instance in instances:
                best_view = instance.get_best_view()
                plt.imshow(best_view.get_image())
                plt.axis("off")
                plt.show()

        t6 = timeit.default_timer()

        if self.robot._rerun and self.update_rerun_every_time:
            self.update_rerun()

        t7 = timeit.default_timer()
        if self.debug_update_timing:
            print("Update timing:")
            print("Time to get observation:", t1 - t0, "seconds, % =", (t1 - t0) / (t7 - t0))
            print("Time to predict:", t2 - t1, "seconds, % =", (t2 - t1) / (t7 - t0))
            print("Time to add obs:", t3 - t2, "seconds, % =", (t3 - t2) / (t7 - t0))
            print("Time to update scene graph:", t4 - t3, "seconds, % =", (t4 - t3) / (t7 - t0))
            print("Time to get 2d map:", t5 - t4, "seconds, % =", (t5 - t4) / (t7 - t0))
            print("Time to debug instances:", t6 - t5, "seconds, % =", (t6 - t5) / (t7 - t0))
            print("Time to update rerun:", t7 - t6, "seconds, % =", (t7 - t6) / (t7 - t0))

    def update_rerun(self):
        """Update the rerun server with the latest observations."""
        if self.robot._rerun:
            self.robot._rerun.update_voxel_map(self.space)
            if self.use_scene_graph:
                self.robot._rerun.update_scene_graph(self.scene_graph, self.semantic_sensor)

        else:
            logger.error("No rerun server available!")

    def _update_scene_graph(self):
        """Update the scene graph with the latest observations."""
        if self.scene_graph is None:
            self.scene_graph = SceneGraph(self.parameters, self.get_voxel_map().get_instances())
        else:
            self.scene_graph.update(self.get_voxel_map().get_instances())
        # For debugging - TODO delete this code
        self.scene_graph.get_relationships(debug=False)
        # self.robot._rerun.update_scene_graph(self.scene_graph, self.semantic_sensor)

    def get_scene_graph(self) -> SceneGraph:
        """Return scene graph, such as it is."""
        self._update_scene_graph()
        return self.scene_graph

    @property
    def manipulation_radius(self) -> float:
        """Return the manipulation radius"""
        return self._manipulation_radius

    def plan_to_instance_for_manipulation(
        self,
        instance: Union[Instance, int],
        start: np.ndarray,
        max_tries: int = 100,
        verbose: bool = True,
        use_cache: bool = False,
    ) -> PlanResult:
        """Move to a specific instance. Goes until a motion plan is found. This function is specifically for manipulation tasks: it plans to a rotation that's rotated 90 degrees from the default.

        Args:
            instance(Instance): an object in the world
            start(np.ndarray): the start position
            max_tries(int): the maximum number of tries to find a plan
            verbose(bool): extra info is printed
            use_cache(bool): whether to use cached plans

        Returns:
            PlanResult: the result of the motion planner
        """

        if isinstance(instance, int):
            _instance = self.get_voxel_map().get_instance(instance)
        else:
            _instance = instance

        return self.plan_to_instance(
            _instance,
            start=start,
            rotation_offset=np.pi / 2,
            radius_m=self._manipulation_radius,
            max_tries=max_tries,
            verbose=verbose,
            use_cache=use_cache,
        )

    def within_reach_of(
        self, instance: Instance, start: Optional[np.ndarray] = None, verbose: bool = False
    ) -> bool:
        """Check if the instance is within manipulation range.

        Args:
            instance(Instance): an object in the world
            start(np.ndarray): the start position
            verbose(bool): extra info is printed

        Returns:
            bool: whether the instance is within manipulation range
        """

        start = start if start is not None else self.robot.get_base_pose()
        if isinstance(start, np.ndarray):
            start = torch.tensor(start, device=self.get_voxel_map().device)
        object_xyz = instance.get_center()
        if np.linalg.norm(start[:2] - object_xyz[:2]) < self._manipulation_radius:
            return True
        if (
            ((instance.point_cloud[:, :2] - start[:2]).norm(dim=-1) < self._manipulation_radius)
            .any()
            .item()
        ):
            return True
        return False

    def plan_to_instance(
        self,
        instance: Instance,
        start: np.ndarray,
        verbose: bool = False,
        max_tries: int = 10,
        radius_m: float = 0.5,
        rotation_offset: float = 0.0,
        use_cache: bool = False,
    ) -> PlanResult:
        """Move to a specific instance. Goes until a motion plan is found.

        Args:
            instance(Instance): an object in the world
            verbose(bool): extra info is printed
            instance_ind(int): if >= 0 we will try to use this to retrieve stored plans
            rotation_offset(float): added to rotation of goal to change final relative orientation
        """

        instance_id = instance.global_id

        if self._realtime_updates:
            if use_cache:
                logger.warning("Cannot use cache with real-time updates")
            use_cache = False

        res = None
        if verbose:
            for j, view in enumerate(instance.instance_views):
                print(f"- instance {instance_id} view {j} at {view.cam_to_world}")

        start_is_valid = self.space.is_valid(start, verbose=False)
        if not start_is_valid:
            return PlanResult(success=False, reason="invalid start state")

        # plan to the sampled goal
        has_plan = False
        if use_cache and instance_id >= 0 and instance_id in self._cached_plans:
            res = self._cached_plans[instance_id]
            has_plan = res.success
            if verbose:
                for j, view in enumerate(instance.instance_views):
                    print(f"- instance {instance_id} view {j} at {view.cam_to_world}")

        # Plan to the instance
        if not has_plan:
            print(
                "planning to instance: ",
                instance_id,
                "max_tries: ",
                max_tries,
                "radius: ",
                radius_m,
                "rotation_offset: ",
                rotation_offset,
            )
            # Call planner
            res = self.plan_to_bounds(
                instance.bounds,
                start,
                verbose,
                max_tries,
                radius_m,
                rotation_offset=rotation_offset,
            )
            if instance_id >= 0:
                self._cached_plans[instance_id] = res

        # Finally, return plan result
        return res

    def plan_to_bounds(
        self,
        bounds: np.ndarray,
        start: np.ndarray,
        verbose: bool = False,
        max_tries: int = 10,
        radius_m: float = 0.5,
        rotation_offset: float = 0.0,
    ) -> PlanResult:
        """Move to be near a bounding box in the world. Goes until a motion plan is found or max_tries is reached.

        Parameters:
            bounds(np.ndarray): the bounding box to move to
            start(np.ndarray): the start position
            verbose(bool): extra info is printed
            max_tries(int): the maximum number of tries to find a plan

        Returns:
            PlanResult: the result of the motion planner
        """

        mask = self.get_voxel_map().mask_from_bounds(bounds)
        try_count = 0
        res = None
        start_is_valid = self.space.is_valid(start, verbose=False)

        conservative_sampling = True
        # We will sample conservatively, staying away from obstacles and the edges of explored space -- at least at first.
        for goal in self.space.sample_near_mask(
            mask,
            radius_m=radius_m,
            conservative=conservative_sampling,
            rotation_offset=rotation_offset,
        ):
            goal = goal.cpu().numpy()
            if verbose:
                print("       Start:", start)
                print("Sampled Goal:", goal)
            show_goal = np.zeros(3)
            show_goal[:2] = goal[:2]
            goal_is_valid = self.space.is_valid(goal, verbose=False)
            if verbose:
                print("Start is valid:", start_is_valid)
                print(" Goal is valid:", goal_is_valid)
            if not goal_is_valid:
                if verbose:
                    print(" -> resample goal.")
                continue

            res = self.planner.plan(start, goal, verbose=False)
            if verbose:
                print("Found plan:", res.success)
            try_count += 1
            if res.success or try_count > max_tries:
                break

        # Planning failed
        if res is None:
            return PlanResult(success=False, reason="no valid plans found")
        return res

    def move_to_instance(self, instance: Instance, max_try_per_instance=10) -> bool:
        """Move to a specific instance. Goes until a motion plan is found.

        Args:
            instance(Instance): an object in the world
            max_try_per_instance(int): number of tries to move to the instance

        Returns:
            bool: whether the robot went to the instance
        """

        print("Moving to instance...")

        # Get the start position
        start = self.robot.get_base_pose()

        # Try to move to the instance
        # We will plan multiple times to see if we can get there
        for i in range(max_try_per_instance):
            res = self.plan_to_instance(instance, start, verbose=True)
            print("Plan result:", res.success)
            if res is not None and res.success:
                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)

                # Follow the planned trajectory
                print("Executing trajectory...")
                self.robot.execute_trajectory(
                    [pt.state for pt in res.trajectory],
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                )

                # Check if the robot is stuck or if anything went wrong during motion execution
                # return self.robot.last_motion_failed() # TODO (hello-atharva): Fix this. This is not implemented
                return True
        return False

    def recover(self):
        # Get current invalid pose

        for step_i in range(5, 100, 5):
            step = step_i / 100.0

            start = self.robot.get_base_pose()

            if self.space.is_valid(start, verbose=True):
                logger.warning("Start state is valid!")
                break

            # Apply relative transformation to XYT
            forward = np.array([-1 * step, 0, 0])
            backward = np.array([step, 0, 0])

            xyt_goal_forward = xyt_base_to_global(forward, start)
            xyt_goal_backward = xyt_base_to_global(backward, start)

            # Is this valid?
            logger.warning(f"Trying to recover with step of {step}...")
            if self.space.is_valid(xyt_goal_backward, verbose=True):
                logger.warning("Trying to move backwards...")
                # Compute the position forward or backward from the robot
                self.robot.move_base_to(xyt_goal_backward, relative=False, blocking=True)
            elif self.space.is_valid(xyt_goal_forward, verbose=True):
                logger.warning("Trying to move forward...")
                # Compute the position forward or backward from the robot
                self.robot.move_base_to(xyt_goal_forward, relative=False, blocking=True)
            else:
                logger.warning(f"Could not recover from invalid start state with step of {step}!")

    def recover_from_invalid_start(self, verbose: bool = True) -> bool:
        """Try to recover from an invalid start state.

        Returns:
            bool: whether the robot recovered from the invalid start state
        """

        print("Recovering from invalid start state...")

        max_tries = 10

        while max_tries > 0:
            self.recover()

            # Get the current position in case we are still invalid
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start, verbose=True)

            if start_is_valid:
                return True

        if not start_is_valid:
            logger.warning("Tried and failed to recover from invalid start state!")

        return start_is_valid

    def move_to_any_instance(
        self, matches: List[Tuple[int, Instance]], max_try_per_instance=10, verbose: bool = False
    ) -> bool:
        """Check instances and find one we can move to"""
        self.current_state = "NAV_TO_INSTANCE"
        self.robot.move_to_nav_posture()
        start = self.robot.get_base_pose()
        start_is_valid = self.space.is_valid(start, verbose=True)
        start_is_valid_retries = 5
        while not start_is_valid and start_is_valid_retries > 0:
            if verbose:
                print(f"Start {start} is not valid. Either back up a bit or go forward.")
            ok = self.recover_from_invalid_start()
            start_is_valid_retries -= 1
            start_is_valid = ok

        res = None

        # Just terminate here - motion planning issues apparently!
        if not start_is_valid:
            return False
            # TODO: fix this
            # raise RuntimeError("Invalid start state!")

        # Find and move to one of these
        for i, match in matches:
            tries = 0
            while tries <= max_try_per_instance:
                print("Checking instance", i)
                # TODO: this is a bad name for this variable
                res = self.plan_to_instance(match, start)
                tries += 1
                if res is not None and res.success:
                    break
                else:
                    # TODO: remove debug code
                    print("-> could not plan to instance", i)
                    if i not in self._object_attempts:
                        self._object_attempts[i] = 1
                    else:
                        self._object_attempts[i] += 1

                    print("no plan found, explore more")
                    self.run_exploration(
                        manual_wait=False,
                        explore_iter=10,
                        task_goal=None,
                        go_home_at_end=False,
                    )
            if res is not None and res.success:
                break

        if res is not None and res.success:
            # Now move to this location
            print("Full plan to object:")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)
            self.robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
            )

            if self.robot.last_motion_failed():
                # We have a problem!
                self.recover_from_invalid_start()

            time.sleep(1.0)
            self.robot.move_base_to([0, 0, np.pi / 2], relative=True)
            self.robot.move_to_manip_posture()
            return True

        return False

    def move_to_manip_posture(self):
        """Move the robot to manipulation posture."""
        self.robot.move_to_manip_posture()

    def move_to_nav_posture(self):
        """Move the robot to navigation posture."""
        self.robot.move_to_nav_posture()

    def print_found_classes(self, goal: Optional[str] = None):
        """Helper. print out what we have found according to detic."""
        if self.semantic_sensor is None:
            logger.warning("Tried to print classes without semantic sensor!")
            return

        instances = self.get_voxel_map().get_instances()
        if goal is not None:
            print(f"Looking for {goal}.")
        print("So far, we have found these classes:")
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            print(i, name, instance.score)

    def start(
        self,
        goal: Optional[str] = None,
        visualize_map_at_start: bool = False,
        can_move: bool = True,
        verbose: bool = True,
    ) -> None:

        # Call the robot's own startup hooks
        started = self.robot.start()
        if not started:
            # update here
            raise RuntimeError("Robot failed to start!")

        if can_move:
            # First, open the gripper...
            self.robot.switch_to_manipulation_mode()
            self.robot.open_gripper()

            # Tuck the arm away
            if verbose:
                print("Sending arm to home...")
            self.robot.move_to_nav_posture()
            if verbose:
                print("... done.")

        # Move the robot into navigation mode
        self.robot.switch_to_navigation_mode()
        if verbose:
            print("- Update map after switching to navigation posture")

        # Add some debugging stuff - show what 3d point clouds look like
        if visualize_map_at_start:
            if not self._realtime_updates:
                self.update(visualize_map=False)  # Append latest observations
            print("- Visualize map after updating")
            self.get_voxel_map().show(
                orig=np.zeros(3),
                xyt=self.robot.get_base_pose(),
                footprint=self.robot.get_robot_model().get_footprint(),
                instances=self.semantic_sensor is not None,
            )
            self.print_found_classes(goal)

    def get_found_instances_by_class(
        self, goal: Optional[str], threshold: int = 0, debug: bool = False
    ) -> List[Tuple[int, Instance]]:
        """Check to see if goal is in our instance memory or not. Return a list of everything with the correct class.

        Parameters:
            goal(str): optional name of the object we want to find
            threshold(int): number of object attempts we are allowed to do for this object
            debug(bool): print debug info

        Returns:
            instance_id(int): a unique int identifying this instance
            instance(Instance): information about a particular object we believe exists
        """
        matching_instances = []
        if goal is None:
            # No goal means no matches
            return []
        instances = self.get_voxel_map().get_instances()
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            if name.lower() == goal.lower():
                matching_instances.append((i, instance))
        return self.filter_matches(matching_instances, threshold=threshold)

    def extract_symbolic_spatial_info(self, instances: List[Instance], debug=False):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        scene_graph = SceneGraph(parameters=self.parameters, instances=instances)
        return scene_graph.get_relationships()

    def get_all_reachable_instances(self, current_pose=None) -> List[Instance]:
        """get all reachable instances with their ids and cache the motion plans.

        Parameters:
            current_pose(np.ndarray): the current pose of the robot

        Returns:
            reachable_matches: list of instances that are reachable from the current pose
        """
        reachable_matches = []
        start = self.robot.get_base_pose() if current_pose is None else current_pose
        for i, instance in enumerate(self.get_voxel_map().get_instances()):
            if i not in self._cached_plans.keys():
                res = self.plan_to_instance(instance, start)
                self._cached_plans[i] = res
            else:
                res = self._cached_plans[i]
            if res.success:
                reachable_matches.append(instance)
        return reachable_matches

    def get_ranked_instances(
        self, goal: str, threshold: int = 0, debug: bool = False
    ) -> List[Tuple[float, int, Instance]]:
        """Get instances and rank them by similarity to the goal, using our encoder, whatever that is.

        Parameters:
            goal(str): optional name of the object we want to find
            threshold(int): number of object attempts we are allowed to do for this object
            debug(bool): print debug info

        Returns:
            ranked_matches: list of tuples with three members:
            score(float): a similarity score between the goal and this instance
            instance_id(int): a unique int identifying this instance
            instance(Instance): information about a particular object detection
        """
        instances = self.get_voxel_map().get_instances()
        goal_emb = self.encode_text(goal)
        ranked_matches = []
        for i, instance in enumerate(instances):
            img_emb = instance.get_image_embedding(
                aggregation_method="mean", normalize=self.normalize_embeddings
            ).to(goal_emb.device)
            score = torch.matmul(goal_emb, img_emb.T).item()
            ranked_matches.append((score, i, instance))
        ranked_matches.sort(reverse=True)
        return ranked_matches

    def get_reachable_instances_by_class(
        self, goal: Optional[str], threshold: int = 0, debug: bool = False
    ) -> List[Instance]:
        """See if we can reach dilated object masks for different objects.

        Parameters:
            goal(str): optional name of the object we want to find
            threshold(int): number of object attempts we are allowed to do for this object
            debug(bool): print debug info

        Returns list of tuples with two members:
            instance_id(int): a unique int identifying this instance
            instance(Instance): information about a particular object we believe exists
        """
        matches = self.get_found_instances_by_class(goal=goal, threshold=threshold, debug=debug)
        reachable_matches = []
        self._cached_plans = {}
        start = self.robot.get_base_pose()
        for i, instance in matches:
            # compute its mask
            # see if this mask's area is explored and reachable from the current robot
            if self.guarantee_instance_is_reachable:
                res = self.plan_to_instance(instance, start)
                self._cached_plans[i] = res
                if res.success:
                    reachable_matches.append(instance)
            else:
                reachable_matches.append(instance)
        return reachable_matches

    def filter_matches(
        self, matches: List[Tuple[int, Instance]], threshold: int = 1
    ) -> List[Tuple[int, Instance]]:
        """return only things we have not tried {threshold} times.

        Parameters:
            matches: list of tuples with two members:
                instance_id(int): a unique int identifying this instance
                instance(Instance): information about a particular object we believe exists
            threshold(int): number of object attempts we are allowed to do for this object

        Returns:
            filtered_matches: list of tuples with two members:
                instance_id(int): a unique int identifying this instance
                instance(Instance): information about a particular object we believe exists
        """
        filtered_matches = []
        for i, instance in matches:
            if i not in self._object_attempts or self._object_attempts[i] < threshold:
                filtered_matches.append((i, instance))
        return filtered_matches

    def go_home(self):
        """Simple helper function to send the robot home safely after a trial. This will use the current map and motion plan all the way there. This is a blocking call, so it will return when the robot is at home.

        If the robot is not able to plan to home, it will print a message and return without moving the robot.

        If the start state is invalid, it will try to recover from the invalid start state before planning to home. If it fails to recover, it will print a message and return without moving the robot.
        """
        print("Go back to (0, 0, 0) to finish...")
        print("- change posture and switch to navigation mode")
        self.current_state = "NAV_TO_HOME"
        self.robot.move_to_nav_posture()

        print("- try to motion plan there")
        start = self.robot.get_base_pose()
        goal = np.array([0, 0, 0])
        print(f"- Current pose is valid: {self.space.is_valid(self.robot.get_base_pose())}")
        print(f"-   start pose is valid: {self.space.is_valid(start)}")
        print(f"-    goal pose is valid: {self.space.is_valid(goal)}")

        # If the start is invalid, try to recover
        if not self.space.is_valid(start):
            print("Start is not valid. Trying to recover...")
            ok = self.recover_from_invalid_start()
            if not ok:
                print("Failed to recover from invalid start state!")
                return

        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            print("Went home!")
        else:
            print("Can't go home; planning failed!")

    def choose_best_goal_instance(self, goal: str, debug: bool = False) -> Instance:
        """Choose the best instance to move to based on the goal. This is done by comparing the goal to the embeddings of the instances in the world. The instance with the highest score is returned. If debug is true, we also print out the scores for the goal and two negative examples. These examples are:
        - "the color purple"
        - "a blank white wall"

        Args:
            goal(str): the goal to move to
            debug(bool): whether to print out debug information

        Returns:
            Instance: the best instance to move to
        """
        instances = self.get_voxel_map().get_instances()
        goal_emb = self.encode_text(goal)
        if debug:
            neg1_emb = self.encode_text("the color purple")
            neg2_emb = self.encode_text("a blank white wall")
        best_instance = None
        best_score = -float("Inf")
        for instance in instances:
            if debug:
                print("# views =", len(instance.instance_views))
                print("    cls =", instance.category_id)
            # TODO: remove debug code when not needed for visualization
            # instance._show_point_cloud_open3d()
            img_emb = instance.get_image_embedding(
                aggregation_method="mean", normalize=self.normalize_embeddings
            )
            goal_score = self.compare_embeddings(goal_emb, img_emb)
            if debug:
                neg1_score = self.compare_embeddings(neg1_emb, img_emb)
                neg2_score = self.compare_embeddings(neg2_emb, img_emb)
                print("scores =", goal_score, neg1_score, neg2_score)
            if goal_score > best_score:
                best_instance = instance
                best_score = goal_score
        return best_instance

    def set_allowed_radius(self, radius: float):
        """Set the allowed radius for the robot to move to. This is used to limit the robot's movement, particularly when exploring.

        Args:
            radius(float): the radius in meters
        """
        logger.info("[Agent] Setting allowed radius to", radius, "meters.")
        self._allowed_radius = radius

        self.get_voxel_map().set_allowed_radius(radius, origin=self.robot.get_base_pose()[0:2])

    def plan_to_frontier(
        self,
        start: np.ndarray,
        manual_wait: bool = False,
        random_goals: bool = False,
        try_to_plan_iter: int = 10,
        fix_random_seed: bool = False,
        verbose: bool = False,
        push_locations_to_stack: bool = False,
    ) -> PlanResult:
        """Motion plan to a frontier location. This is a location that is on the edge of the explored space. We use the voxel grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            start(np.ndarray): the start position
            manual_wait(bool): whether to wait for user input
            random_goals(bool): whether to sample random goals
            try_to_plan_iter(int): the number of tries to find a plan
            fix_random_seed(bool): whether to fix the random seed
            verbose(bool): extra info is printed
            push_locations_to_stack(bool): whether to push visited locations to planning stack. can help get you unstuck, maybe?

        Returns:
            PlanResult: the result of the motion planner
        """
        start_is_valid = self.space.is_valid(start, verbose=True)
        # if start is not valid move backwards a bit
        if not start_is_valid:
            if verbose:
                print("Start not valid. Try to recover.")
            ok = self.recover_from_invalid_start()
            if not ok:
                return PlanResult(False, reason="invalid start state")
        else:
            print("Planning to frontier...")

        # sample a goal
        if random_goals:
            goal = next(self.space.sample_random_frontier()).cpu().numpy()
        else:
            # Get frontier sampler
            sampler = self.space.sample_closest_frontier(
                start,
                verbose=False,
                expand_size=self._default_expand_frontier_size,
                min_dist=self._frontier_min_dist,
                step_dist=self._frontier_step_dist,
            )
            for i, goal in enumerate(sampler):
                if goal is None:
                    # No more positions to sample
                    if verbose:
                        print("No more frontiers to sample.")
                    break

                # print("Sampled Goal is:", goal.cpu().numpy())
                if self._previous_goal is not None:
                    if np.linalg.norm(goal.cpu().numpy() - self._previous_goal) < 0.1:
                        if verbose:
                            print("Same goal as last time. Skipping.")
                        self._previous_goal = goal.cpu().numpy()
                        continue
                self._previous_goal = goal.cpu().numpy()

                print("Checking if goal is valid...")
                if self.space.is_valid(goal.cpu().numpy(), verbose=True):
                    if verbose:
                        print("       Start:", start)
                        print("Sampled Goal:", goal.cpu().numpy())
                else:
                    continue

                # For debugging, we can set the random seed to 0
                if fix_random_seed:
                    np.random.seed(0)
                    random.seed(0)

                if push_locations_to_stack:
                    print("Pushing visited locations to stack...")
                    self.planner.space.push_locations_to_stack(self.get_history(reversed=True))
                # print("Planning to goal...")
                res = self.planner.plan(start, goal.cpu().numpy())
                if res.success:
                    if verbose:
                        print("Plan successful!")
                    return res
                else:
                    # print("Plan failed. Reason:", res.reason)
                    if verbose:
                        print("Plan failed. Reason:", res.reason)
        return PlanResult(False, reason="no valid plans found")

    def get_history(self, reversed: bool = False) -> List[np.ndarray]:
        """Get the history of the robot's positions."""
        history = []
        for obs in self.get_voxel_map().observations:
            history.append(obs.base_pose)
        if reversed:
            history.reverse()
        return history

    def run_exploration(
        self,
        manual_wait: bool = False,
        explore_iter: int = 3,
        try_to_plan_iter: int = 10,
        dry_run: bool = False,
        random_goals: bool = False,
        visualize: bool = False,
        task_goal: str = None,
        go_home_at_end: bool = False,
        show_goal: bool = False,
        audio_feedback: bool = False,
    ) -> Optional[List[Instance]]:
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        self.current_state = "EXPLORE"
        self.robot.move_to_nav_posture()
        all_starts = []
        all_goals: List[List] = []

        if audio_feedback:
            self.robot.say("Starting exploration!")
            time.sleep(3.0)

        # Explore some number of times
        matches = []
        no_success_explore = True
        rotated = False
        for i in range(explore_iter):
            if audio_feedback:
                self.robot.say_sync(f"Step {i + 1} of {explore_iter}")
                time.sleep(2.0)

            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start, verbose=True)
            # if start is not valid move backwards a bit
            if not start_is_valid:
                print("Start not valid. back up a bit.")
                if audio_feedback:
                    self.robot.say_sync("Start not valid. Backing up a bit.")
                    time.sleep(2.0)

                ok = self.recover_from_invalid_start()
                if ok:
                    start = self.robot.get_base_pose()
                    start_is_valid = self.space.is_valid(start, verbose=True)
                if not start_is_valid:
                    print("Failed to recover from invalid start state!")

                    if audio_feedback:
                        self.robot.say_sync("Failed to recover from invalid start state!")
                        time.sleep(2.0)
                    break

            # Now actually plan to the frontier
            print("       Start:", start)
            self.print_found_classes(task_goal)

            if audio_feedback and self.semantic_sensor is not None:
                if len(all_goals) > 0:
                    self.robot.say_sync(
                        f"So far, I have found {len(all_goals)} object{'s' if len(all_goals) > 2 else ''}"
                    )
                else:
                    self.robot.say_sync(f"My map is currently empty")
                time.sleep(1.0)

            if audio_feedback:
                self.robot.say_sync("Looking for frontiers nearby...")
                time.sleep(1.0)

            res = self.plan_to_frontier(
                start=start, random_goals=random_goals, try_to_plan_iter=try_to_plan_iter
            )

            # if it succeeds, execute a trajectory to this position
            if res.success:
                rotated = False
                no_success_explore = False
                print("Exploration plan to frontier successful!")

                if audio_feedback:
                    self.robot.say_sync("I found a frontier to explore. Adding it as a goal.")
                    time.sleep(4.0)

                for i, pt in enumerate(res.trajectory):
                    print(i, pt.state)
                all_starts.append(start)
                all_goals.append(res.trajectory[-1].state)
                if visualize:
                    print("Showing goal location:")
                    robot_center = np.zeros(3)
                    robot_center[:2] = self.robot.get_base_pose()[:2]
                    self.get_voxel_map().show(
                        orig=robot_center,
                        xyt=res.trajectory[-1].state,
                        footprint=self.robot.get_robot_model().get_footprint(),
                        instances=self.semantic_sensor is not None,
                    )
                if not dry_run:

                    if audio_feedback:
                        self.robot.say_sync("Attempting to move to this goal.")
                        time.sleep(4.0)

                    self.robot.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )

            else:
                # Try rotating in place if we can't find a decent frontier to get to
                if not rotated:
                    if audio_feedback:
                        self.robot.say("No frontier found. Trying to rotate in place.")
                    self.rotate_in_place()
                    rotated = True
                    continue

                if rotated:
                    if audio_feedback:
                        self.robot.say("No frontier found. We're done exploring!")
                    print("No where left to explore. Quitting.")
                    break

                # if it fails, try again or just quit
                if self._retry_on_fail:
                    print("Exploration failed. Try again!")

                    if audio_feedback:
                        self.robot.say_sync("Exploration failed. Trying again.")
                        time.sleep(4.0)
                    continue
                else:
                    print("Start = ", start)
                    print("Reason = ", res.reason)
                    print("Exploration failed. Quitting!")

                    if audio_feedback:
                        self.robot.say_sync("Exploration failed. Quitting.")
                        time.sleep(4.0)
                    break

            # Append latest observations

            if audio_feedback:
                self.robot.say_sync("Recording new observations.")
                time.sleep(4.0)

            if not self._realtime_updates:
                self.update()

            # self.save_svm("", filename=f"debug_svm_{i:03d}.pkl")
            if visualize:
                # After doing everything - show where we will move to
                robot_center = np.zeros(3)
                robot_center[:2] = self.robot.get_base_pose()[:2]
                self.navigation_space.show(
                    orig=robot_center,
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                )

            if manual_wait:
                input("... press enter ...")

            if task_goal is not None:
                matches = self.get_reachable_instances_by_class(task_goal)
                if len(matches) > 0:
                    print("!!! GOAL FOUND! Done exploration. !!!")

                    if audio_feedback:
                        self.robot.say_sync("Goal found! Done exploration.")
                        time.sleep(4.0)
                    break

        if go_home_at_end:
            self.current_state = "NAV_TO_HOME"
            # Finally - plan back to (0,0,0)
            print("Go back to (0, 0, 0) to finish...")

            if audio_feedback:
                self.robot.say_sync("Trying to go back to my initial position.")
                time.sleep(4.0)

            start = self.robot.get_base_pose()
            goal = np.array([0, 0, 0])
            self.planner.space.push_locations_to_stack(self.get_history(reversed=True))
            res = self.planner.plan(start, goal)
            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Full plan to home:")

                if audio_feedback:
                    self.robot.say_sync("I found a plan to home.")
                    time.sleep(4.0)

                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)
                if not dry_run:
                    self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            else:
                print("WARNING: planning to home failed!")

        return matches

    def show_voxel_map(self):
        """Show the voxel map in Open3D for debugging."""
        self.get_voxel_map().show(
            orig=np.zeros(3),
            xyt=self.robot.get_base_pose(),
            footprint=self.robot.get_robot_model().get_footprint(),
            instances=self.semantic_sensor is not None,
            planner_visuals=True,
        )

    def move_closed_loop(self, goal: np.ndarray, max_time: float = 10.0) -> bool:
        """Helper function which will move while also checking which position the robot reached.

        Args:
            goal(np.ndarray): the goal position
            max_time(float): the maximum time to wait for the robot to reach the goal

        Returns:
            bool: true if the robot reached the goal, false otherwise
        """
        t0 = timeit.default_timer()
        self.robot.move_to_nav_posture()
        while True:
            self.robot.move_base_to(goal, blocking=False, timeout=30.0)
            if self.robot.last_motion_failed():
                return False
            position = self.robot.get_base_pose()
            if (
                np.linalg.norm(position[:2] - goal[:2]) < 0.1
                and angle_difference(position[2], goal[2]) < 0.1
                and self.robot.at_goal()
            ):
                return True
            t1 = timeit.default_timer()
            if t1 - t0 > max_time:
                return False
            time.sleep(0.1)

    def reset(self, verbose: bool = True):
        """Reset the robot's spatial memory. This deletes the instance memory and spatial map, and clears all observations.

        Args:
            verbose(bool): print out a message to the user making sure this does not go unnoticed. Defaults to True.
        """
        if verbose:
            print(
                "[WARNING] Resetting the robot's spatial memory. Everything it knows will go away!"
            )
        with self._voxel_map_lock:
            self.voxel_map.reset()
        self.reset_object_plans()

        if self._realtime_updates:
            # Clear the observations
            with self._obs_history_lock:
                self.obs_history.clear()
                self.obs_count = 0
                self._matched_observations_poses.clear()
                self._matched_vertices_obs_count.clear()

    def save_instance_images(self, root: Union[Path, str] = ".", verbose: bool = False) -> None:
        """Save out instance images from the voxel map that we have collected while exploring."""

        if isinstance(root, str):
            root = Path(root)

        # Write out instance images
        for i, instance in enumerate(self.get_voxel_map().get_instances()):
            for j, view in enumerate(instance.instance_views):
                image = Image.fromarray(view.cropped_image.byte().cpu().numpy())
                filename = f"instance{i}_view{j}.png"
                if verbose:
                    print(f"Saving instance image {i} view {j} to {root / filename}")
                image.save(root / filename)

    # OVMM primitives
    def go_to(self, object_goal: str, **kwargs) -> bool:
        """Go to a specific object in the world."""
        # Find closest instance
        print(f"Going to {object_goal}")
        _, instance = self.get_instance_from_text(object_goal)
        instance.show_best_view(title=object_goal)

        # Move to the instance
        return self.move_to_instance(instance)

    def pick(self, object_goal: str, **kwargs) -> bool:
        """Pick up an object."""
        # Find closest instance
        instances = self.get_found_instances_by_class(object_goal)
        if len(instances) == 0:
            return False

        # Move to the instance with the highest score
        # Sort by score
        instances.sort(key=lambda x: x[1].score, reverse=True)
        instance = instances[0][1]

        # Move to the instance
        if not self.move_to_instance(instance):
            return False

        # Grasp the object
        return self.grasp_object(object_goal)

    def place(self, object_goal: str, **kwargs) -> bool:
        """Place an object."""
        # Find closest instance
        instances = self.get_found_instances_by_class(object_goal)

        if len(instances) == 0:
            return False

        # Move to the instance with the highest score
        # Sort by score
        instances.sort(key=lambda x: x[1].score, reverse=True)
        instance = instances[0][1]

        # Move to the instance
        if not self.move_to_instance(instance):
            return False

        # Place the object
        return self.place_object(object_goal)

    def say(self, msg: str):
        """Provide input either on the command line or via chat client"""
        self.tts.say_async(msg)

    def robot_say(self, msg: str):
        """Have the robot say something out loud. This will send the text over to the robot from wherever the client is running."""
        msg = msg.strip('"' + "'")
        self.robot.say(msg)

    def ask(self, msg: str) -> str:
        """Receive input from the user either via the command line or something else"""
        # if self.chat is not None:
        #  return self.chat.input(msg)
        # else:
        # TODO: support other ways of saying
        return input(msg)

    def open_cabinet(self, object_goal: str, **kwargs) -> bool:
        """Open a cabinet."""
        print("Not implemented yet.")
        return True

    def close_cabinet(self, object_goal: str, **kwargs) -> bool:
        """Close a cabinet."""
        print("Not implemented yet.")
        return True

    def wave(self, **kwargs) -> bool:
        """Wave."""
        n_waves = 3
        pitch = 0.2
        yaw_amplitude = 0.25
        roll_amplitude = 0.15
        lift_height = 1.0
        self.robot.switch_to_manipulation_mode()

        first_pose = [0.0, lift_height, 0.05, 0.0, 0.0, 0.0]

        # move to initial lift height
        first_pose[1] = lift_height
        self.robot.arm_to(first_pose, blocking=True)

        # generate poses
        wave_poses = np.zeros((n_waves * 2, 6))
        for i in range(n_waves):
            j = i * 2
            wave_poses[j] = [0.0, lift_height, 0.05, -yaw_amplitude, pitch, -roll_amplitude]
            wave_poses[j + 1] = [0.0, lift_height, 0.05, yaw_amplitude, pitch, roll_amplitude]

        # move to poses w/o blocking to make smoother motions
        for pose in wave_poses:
            self.robot.arm_to(pose, blocking=False)
            time.sleep(0.375)

        self.robot.switch_to_navigation_mode()
        return True

    def load_map(
        self,
        filename: str,
        color_weight: float = 0.5,
        debug: bool = False,
        ransac_distance_threshold: float = 0.1,
    ) -> None:
        """Load a map from a PKL file. Creates a new voxel map and loads the data from the file into this map. Then uses RANSAC to figure out where the current map and the loaded map overlap, computes a transform, and applies this transform to the loaded map to align it with the current map.

        Args:
            filename(str): the name of the file to load the map from
        """

        # Load the map from the file
        loaded_voxel_map = self._create_voxel_map(self.parameters)
        loaded_voxel_map.read_from_pickle(filename, perception=self.semantic_sensor)

        xyz1, _, _, rgb1 = loaded_voxel_map.get_pointcloud()
        xyz2, _, _, rgb2 = self.get_voxel_map().get_pointcloud()

        # tform = find_se3_transform(xyz1, xyz2, rgb1, rgb2)
        tform, fitness, inlier_rmse, num_inliers = ransac_transform(
            xyz1, xyz2, visualize=debug, distance_threshold=ransac_distance_threshold
        )
        print("------------- Loaded map ---------------")
        print("Aligning maps...")
        print("RANSAC transform from old to new map:")
        print(tform)
        print("Fitness:", fitness)
        print("Inlier RMSE:", inlier_rmse)
        print("Num inliers:", num_inliers)
        print("----------------------------------------")

        # Apply the transform to the loaded map and
        if debug:
            # for visualization
            from stretch.utils.point_cloud import show_point_cloud

            # Apply the transform to the loaded map
            xyz1 = xyz1 @ tform[:3, :3].T + tform[:3, 3]

            _xyz = np.concatenate([xyz1.cpu().numpy(), xyz2.cpu().numpy()], axis=0)
            _rgb = np.concatenate([rgb1.cpu().numpy(), rgb2.cpu().numpy() * 0.5], axis=0) / 255
            show_point_cloud(_xyz, _rgb, orig=np.zeros(3))

        # Add the loaded map to the current map
        print("Reprocessing map with new pose transform...")
        self.get_voxel_map().read_from_pickle(
            filename, perception=self.semantic_sensor, transform_pose=tform
        )
        self.get_voxel_map().show()

    def get_detections(self, **kwargs) -> List[str]:
        """Get the current detections."""
        instances = self.get_voxel_map().get_instances()

        # Consider only instances close to the robot
        robot_pose = self.robot.get_base_pose()
        close_instances = []

        for instance in instances:
            instance_pose = instance.get_best_view().cam_to_world
            distance = np.linalg.norm(robot_pose[:2] - instance_pose[:2])
            if distance < 2.0:
                close_instances.append(instance)

        close_instances_names = [
            self.semantic_sensor.get_class_name_for_id(instance.category_id)
            for instance in close_instances
        ]
        return close_instances_names

    def execute(self, plan: str) -> bool:
        """Execute a plan function given as a string."""
        try:
            exec(plan)

            return True
        except Exception as e:
            print(f"Failed to execute plan: {e}")
            return False

    def get_object_centric_world_representation(
        self,
        instance_memory,
        task: str = None,
        text_features=None,
        max_context_length=20,
        sample_strategy="clip",
        scene_graph=None,
    ) -> ObjectCentricObservations:
        """Get version that LLM can handle - convert images into torch if not already in that format. This will also clip the number of instances to the max context length.

        Args:
            instance_memory: the instance memory
            max_context_length: the maximum number of instances to consider
            sample_strategy: the strategy to use for sampling instances
            task: the task that the robot is trying to solve
            text_features: the text features
            scene_graph: the scene graph

        Returns:
            ObjectCentricObservations: a list of object-centric observations
        """

        obs = ObjectCentricObservations(object_images=[], scene_graph=scene_graph)
        candidate_objects = []
        for instance in instance_memory:
            global_id = instance.global_id
            crop = instance.get_best_view()
            features = crop.embedding
            crop = crop.cropped_image
            if isinstance(crop, np.ndarray):
                crop = torch.from_numpy(crop)
            candidate_objects.append((crop, global_id, features))

        if len(candidate_objects) >= max_context_length:
            logger.warning(
                f"\nWarning: VLMs can only handle limited size of crops -- ignoring instances using strategy: {sample_strategy}..."
            )
            if sample_strategy == "clip":
                # clip ranking
                if task:
                    print(f"clip sampling based on task: {task}")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    image_features = []
                    for obj in candidate_objects:
                        image_features.append(obj[2])
                    image_features = torch.stack(image_features).squeeze(1).to(device)
                    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
                    _, indices = similarity[0].topk(len(candidate_objects))
                    sorted_objects = []
                    for k in indices:
                        sorted_objects.append(candidate_objects[k.item()])
                    candidate_objects = sorted_objects
            else:
                raise NotImplementedError

        for crop_id, obj in enumerate(
            candidate_objects[: min(max_context_length, len(candidate_objects))]
        ):
            obs.object_images.append(
                ObjectImage(crop_id=crop_id, image=obj[0].contiguous(), instance_id=obj[1])
            )
        return obs

    def get_object_centric_observations(
        self,
        show_prompts: bool = False,
        task: Optional[str] = None,
        current_pose=None,
        plan_with_reachable_instances=False,
        plan_with_scene_graph=False,
    ) -> ObjectCentricObservations:
        """Get object-centric observations for the current state of the world. This is a list of images and associated object information.

        Args:
            show_prompts(bool): whether to show prompts to the user
            task(str): the task that the robot is trying to solve
            current_pose(np.ndarray): the current pose of the robot

        Returns:
            ObjectCentricObservations: a list of object-centric observations
        """
        if plan_with_reachable_instances:
            instances = self.get_all_reachable_instances(current_pose=current_pose)
        else:
            instances = self.get_voxel_map().get_instances()

        scene_graph = None

        # Create a scene graph if needed
        if plan_with_scene_graph:
            scene_graph = self.extract_symbolic_spatial_info(instances)

        world_representation = self.get_object_centric_world_representation(
            instances,
            self.parameters.get("vlm/vlm_context_length", 20),
            self.parameters.get("vlm/sample_strategy", "clip"),
            task,
            self.encode_text(task),
            scene_graph,
        )
        return world_representation

    def create_semantic_sensor(self) -> OvmmPerception:
        """Create a semantic sensor."""

        # This is a lazy load since it can be a difficult dependency
        from stretch.perception import create_semantic_sensor

        return create_semantic_sensor(self.parameters)

    def save_map(self, filename: Optional[str] = None) -> None:
        """Save the current map to a file.

        Args:
            filename(str): the name of the file to save the map to
        """
        if filename is None or filename == "":
            filename = memory.get_path_to_saved_map()

        # Backup the saved map
        memory.backup_saved_map()

        self.get_voxel_map().write_to_pickle(filename)
