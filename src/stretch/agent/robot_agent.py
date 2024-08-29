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
import datetime
import random
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import stretch.utils.logger as logger
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractGraspClient, AbstractRobotClient
from stretch.mapping.instance import Instance
from stretch.mapping.scene_graph import SceneGraph
from stretch.mapping.voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace
from stretch.motion import ConfigurationSpace, PlanResult
from stretch.motion.algo import RRTConnect, Shortcut, SimplifyXYT
from stretch.perception.encoders import BaseImageTextEncoder, get_encoder
from stretch.perception.wrapper import OvmmPerception
from stretch.utils.geometry import angle_difference


class RobotAgent:
    """Basic demo code. Collects everything that we need to make this work."""

    _retry_on_fail: bool = False
    debug_update_timing: bool = False
    update_rerun_every_time: bool = False

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        rpc_stub=None,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
    ):
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.rpc_stub = rpc_stub
        self.grasp_client = grasp_client
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.semantic_sensor = semantic_sensor
        self.normalize_embeddings = True
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]
        self.current_state = "WAITING"
        self.encoder = get_encoder(parameters["encoder"], parameters.get("encoder_args", {}))
        self.obs_count = 0
        self.obs_history = []
        self.guarantee_instance_is_reachable = parameters.guarantee_instance_is_reachable
        self.plan_with_reachable_instances = parameters["plan_with_reachable_instances"]
        self.use_scene_graph = parameters["plan_with_scene_graph"]
        self.tts = get_text_to_speech(parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory

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

        if voxel_map is not None:
            self.voxel_map = voxel_map
        else:
            self.voxel_map = self._create_voxel_map(parameters)

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot.get_robot_model(),
            step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters["dilate_frontier_size"],
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
            grid=self.voxel_map.grid,
        )

        # Dictionary storing attempts to visit each object
        self._object_attempts = {}
        self._cached_plans = {}

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Create a simple motion planner
        self.planner = RRTConnect(self.space, self.space.is_valid)
        if parameters["motion_planner"]["shortcut_plans"]:
            self.planner = Shortcut(self.planner, parameters["motion_planner"]["shortcut_iter"])
        if parameters["motion_planner"]["simplify_plans"]:
            self.planner = SimplifyXYT(self.planner, min_step=0.05, max_step=1.0, num_steps=8)

        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    def _create_voxel_map(self, parameters: Parameters) -> SparseVoxelMap:
        """Create a voxel map from parameters.

        Args:
            parameters(Parameters): the parameters for the voxel map

        Returns:
            SparseVoxelMap: the voxel map
        """
        return SparseVoxelMap(
            resolution=self._voxel_size,
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            obs_min_density=parameters["obs_min_density"],
            encoder=self.encoder,
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            use_instance_memory=(self.semantic_sensor is not None or self._use_instance_memory),
            instance_memory_kwargs={
                "min_pixels_for_instance_view": parameters.get("min_pixels_for_instance_view", 100),
                "min_instance_thickness": parameters.get(
                    "instance_memory/min_instance_thickness", 0.01
                ),
                "min_instance_vol": parameters.get("instance_memory/min_instance_vol", 1e-6),
                "max_instance_vol": parameters.get("instance_memory/max_instance_vol", 10.0),
                "min_instance_height": parameters.get("instance_memory/min_instance_height", 0.1),
                "max_instance_height": parameters.get("instance_memory/max_instance_height", 1.8),
                "min_pixels_for_instance_view": parameters.get(
                    "instance_memory/min_pixels_for_instance_view", 100
                ),
                "min_percent_for_instance_view": parameters.get(
                    "instance_memory/min_percent_for_instance_view", 0.2
                ),
                "open_vocab_cat_map_file": parameters.get("open_vocab_category_map_file", None),
                "use_visual_feat": parameters.get("use_visual_feat", False),
            },
            prune_detected_objects=parameters.get("prune_detected_objects", False),
        )

    @property
    def feature_match_threshold(self) -> float:
        """Return the feature match threshold"""
        return self._is_match_threshold

    @property
    def voxel_size(self) -> float:
        """Return the voxel size in meters"""
        return self._voxel_size

    def get_encoder(self) -> BaseImageTextEncoder:
        """Return the encoder in use by this model"""
        return self.encoder

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using the encoder"""
        return self.encoder.encode_text(text)

    def encode_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encode image using the encoder"""
        return self.encoder.encode_image(image)

    def get_instance_from_text(
        self,
        text_query: str,
        aggregation_method: str = "mean",
        normalize: bool = False,
        verbose: bool = True,
    ) -> Optional[Instance]:
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
        encoded_text = self.encode_text(text_query).to(self.voxel_map.device)
        best_instance = None
        best_activation = -1.0
        for instance in self.voxel_map.get_instances():
            ins = instance.get_instance_id()
            emb = instance.get_image_embedding(
                aggregation_method=aggregation_method, normalize=normalize
            )
            activation = torch.cosine_similarity(emb, encoded_text, dim=-1)
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

    def get_instances_from_text(
        self,
        text_query: str,
        aggregation_method: str = "mean",
        normalize: bool = False,
        verbose: bool = True,
        threshold: float = 0.5,
    ) -> List[Tuple[float, Instance]]:
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
        encoded_text = self.encode_text(text_query).to(self.voxel_map.device)
        # Compute the cosine similarity between the text query and each instance embedding
        for instance in self.voxel_map.get_instances():
            ins = instance.get_instance_id()
            emb = instance.get_image_embedding(
                aggregation_method=aggregation_method, normalize=normalize
            )
            activation = torch.cosine_similarity(emb, encoded_text, dim=-1)
            # Add the instance to the list of matches if the cosine similarity is above the threshold
            if activation.item() > threshold:
                activations.append(activation.item())
                matches.append(instance)
                if verbose:
                    print(f" - Instance {ins} has activation {activation.item()}")
            elif verbose:
                print(f" - Skipped instance {ins} with activation {activation.item()}")
        return activations, matches

    def get_navigation_space(self) -> ConfigurationSpace:
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
        self, steps: int = 12, visualize: bool = False, verbose: bool = True
    ) -> bool:
        """Simple helper function to make the robot rotate in place. Do a 360 degree turn to get some observations (this helps debug the robot and create a nice map).

        Args:
            steps(int): number of steps to rotate (each step is 360 degrees / steps). Default is 12.
            visualize(bool): show the map as we rotate. Default is False.

        Returns:
            executed(bool): false if we did not actually do any rotations"""
        logger.info("Rotate in place")
        if steps <= 0:
            return False

        step_size = 2 * np.pi / steps
        i = 0
        x, y, theta = self.robot.get_base_pose()
        if verbose:
            print(f"==== ROTATE IN PLACE at {x}, {y} ====")
        while i < steps - 1:
            t0 = timeit.default_timer()
            self.robot.navigate_to(
                [x, y, theta + ((i + 1) * step_size)],
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
            self.update()
            t1 = timeit.default_timer()
            if verbose:
                print("Update took", t1 - t0, "seconds")

            if visualize:
                self.voxel_map.show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                    instances=self.semantic_sensor is not None,
                )

        return True

    def get_command(self):
        if (
            "command" in self.parameters.data.keys()
        ):  # TODO: this was breaking. Should this be a class method
            return self.parameters["command"]
        else:
            return self.ask("please type any task you want the robot to do: ")

    def show_map(self):
        """Helper function to visualize the 3d map as it stands right now"""
        self.voxel_map.show(
            orig=np.zeros(3),
            xyt=self.robot.get_base_pose(),
            footprint=self.robot.get_robot_model().get_footprint(),
            instances=self.semantic_sensor is not None,
        )

    def update(self, visualize_map: bool = False, debug_instances: bool = False):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = None
        t0 = timeit.default_timer()

        while obs is None:
            obs = self.robot.get_observation()
            t1 = timeit.default_timer()
            if t1 - t0 > 10:
                logger.error("Failed to get observation")
                return

        t1 = timeit.default_timer()
        self.obs_history.append(obs)
        self.obs_count += 1
        # Optionally do this
        if self.semantic_sensor is not None:
            # Semantic prediction
            obs = self.semantic_sensor.predict(obs)

        t2 = timeit.default_timer()
        self.voxel_map.add_obs(obs)

        t3 = timeit.default_timer()

        if self.use_scene_graph:
            self._update_scene_graph()
            self.scene_graph.get_relationships()

        t4 = timeit.default_timer()

        # Add observation - helper function will unpack it
        if visualize_map:
            # Now draw 2d maps to show what was happening
            self.voxel_map.get_2d_map(debug=True)

        t5 = timeit.default_timer()

        if debug_instances:
            # We need to load and configure matplotlib here
            # to make sure that it's set up properly.
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            instances = self.voxel_map.get_instances()
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
        else:
            logger.error("No rerun server available!")

    def _update_scene_graph(self):
        """Update the scene graph with the latest observations."""
        if self.scene_graph is None:
            self.scene_graph = SceneGraph(self.parameters, self.voxel_map.get_instances())
        else:
            self.scene_graph.update(self.voxel_map.get_instances())
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
        instance: Instance,
        start: np.ndarray,
        max_tries: int = 100,
        verbose: bool = True,
        use_cache: bool = True,
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
        return self.plan_to_instance(
            instance,
            start=start,
            rotation_offset=np.pi / 2,
            radius_m=self._manipulation_radius,
            max_tries=max_tries,
            verbose=verbose,
            use_cache=use_cache,
        )

    def plan_to_instance(
        self,
        instance: Instance,
        start: np.ndarray,
        verbose: bool = False,
        instance_id: int = -1,
        max_tries: int = 10,
        radius_m: float = 0.5,
        rotation_offset: float = 0.0,
        use_cache: bool = True,
    ) -> PlanResult:
        """Move to a specific instance. Goes until a motion plan is found.

        Args:
            instance(Instance): an object in the world
            verbose(bool): extra info is printed
            instance_ind(int): if >= 0 we will try to use this to retrieve stored plans
            rotation_offset(float): added to rotation of goal to change final relative orientation
        """

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
                print(f"- try retrieving cached plan for {instance_id}: {has_plan=}")

        if not has_plan:
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

        mask = self.voxel_map.mask_from_bounds(bounds)
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
                print(f"Start {start} is not valid. back up a bit.")
            self.robot.navigate_to([-0.1, 0, 0], relative=True)
            # Get the current position in case we are still invalid
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start, verbose=True)
            start_is_valid_retries -= 1
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
                res = self.plan_to_instance(match, start, instance_id=i)
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
                        5,  # TODO: pass rate into parameters
                        False,  # TODO: pass manual_wait into parameters
                        explore_iter=10,
                        task_goal=None,
                        go_home_at_end=False,  # TODO: pass into parameters
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
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("ROBOT IS STUCK! Move back!")
                r = np.random.randint(3)
                if r == 0:
                    self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                elif r == 1:
                    self.robot.navigate_to([0, 0, np.pi / 4], relative=True, blocking=True)
                elif r == 2:
                    self.robot.navigate_to([0, 0, -np.pi / 4], relative=True, blocking=True)
                return False

            time.sleep(1.0)
            self.robot.navigate_to([0, 0, np.pi / 2], relative=True)
            self.robot.move_to_manip_posture()
            return True

        return False

    def print_found_classes(self, goal: Optional[str] = None):
        """Helper. print out what we have found according to detic."""
        if self.semantic_sensor is None:
            logger.warning("Tried to print classes without semantic sensor!")
            return

        instances = self.voxel_map.get_instances()
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
        verbose: bool = False,
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
                print("Sending arm to  home...")
            self.robot.move_to_nav_posture()
            if verbose:
                print("... done.")

        # Move the robot into navigation mode
        self.robot.switch_to_navigation_mode()
        if verbose:
            print("- Update map after switching to navigation posture")

        # Add some debugging stuff - show what 3d point clouds look like
        if visualize_map_at_start:
            self.update(visualize_map=False)  # Append latest observations
            print("- Visualize map after updating")
            self.voxel_map.show(
                orig=np.zeros(3),
                xyt=self.robot.get_base_pose(),
                footprint=self.robot.get_robot_model().get_footprint(),
                instances=self.semantic_sensor is not None,
            )
            self.print_found_classes(goal)

    def encode_text(self, text: str):
        """Helper function for getting text embeddings"""
        emb = self.encoder.encode_text(text)
        if self.normalize_embeddings:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

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
        instances = self.voxel_map.get_instances()
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            if name.lower() == goal.lower():
                matching_instances.append((i, instance))
        return self.filter_matches(matching_instances, threshold=threshold)

    def extract_symbolic_spatial_info(self, instances, debug=False):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        scene_graph = SceneGraph(instances)
        return scene_graph.get_relationships()

    def get_all_reachable_instances(self, current_pose=None) -> List[Tuple[int, Instance]]:
        """get all reachable instances with their ids and cache the motion plans"""
        reachable_matches = []
        start = self.robot.get_base_pose() if current_pose is None else current_pose
        for i, instance in enumerate(self.voxel_map.get_instances()):
            if i not in self._cached_plans.keys():
                res = self.plan_to_instance(instance, start, instance_id=i)
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
        instances = self.voxel_map.get_instances()
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
    ) -> List[Tuple[int, Instance]]:
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
                res = self.plan_to_instance(instance, start, instance_id=i)
                self._cached_plans[i] = res
                if res.success:
                    reachable_matches.append(instance)
            else:
                reachable_matches.append(instance)
        return reachable_matches

    def filter_matches(
        self, matches: List[Tuple[int, Instance]], threshold: int = 1
    ) -> Tuple[int, Instance]:
        """return only things we have not tried {threshold} times"""
        filtered_matches = []
        for i, instance in matches:
            if i not in self._object_attempts or self._object_attempts[i] < threshold:
                filtered_matches.append((i, instance))
        return filtered_matches

    def go_home(self):
        """Simple helper function to send the robot home safely after a trial. This will use the current map and motion plan all the way there."""
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
        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            print("Done!")
        else:
            print("Can't go home; planning failed!")

    def choose_best_goal_instance(self, goal: str, debug: bool = False) -> Instance:
        instances = self.voxel_map.get_instances()
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
            goal_score = torch.matmul(goal_emb, img_emb).item()
            if debug:
                neg1_score = torch.matmul(neg1_emb, img_emb).item()
                neg2_score = torch.matmul(neg2_emb, img_emb).item()
                print("scores =", goal_score, neg1_score, neg2_score)
            if goal_score > best_score:
                best_instance = instance
                best_score = goal_score
        return best_instance

    def plan_to_frontier(
        self,
        start: np.ndarray,
        rate: int = 10,
        manual_wait: bool = False,
        random_goals: bool = False,
        try_to_plan_iter: int = 10,
        fix_random_seed: bool = False,
        verbose: bool = False,
    ) -> PlanResult:
        """Motion plan to a frontier location."""
        start_is_valid = self.space.is_valid(start, verbose=True)
        # if start is not valid move backwards a bit
        if not start_is_valid:
            if verbose:
                print("Start not valid. back up a bit.")
            return PlanResult(False, reason="invalid start state")

        # sample a goal
        if random_goals:
            goal = next(self.space.sample_random_frontier()).cpu().numpy()
        else:
            # Get frontier sampler
            sampler = self.space.sample_closest_frontier(
                start,
                verbose=False,
                min_dist=self._frontier_min_dist,
                step_dist=self._frontier_step_dist,
            )
            for i, goal in enumerate(sampler):
                if goal is None:
                    # No more positions to sample
                    break

                # For debugging, we can set the random seed to 0
                if fix_random_seed:
                    np.random.seed(0)
                    random.seed(0)

                res = self.planner.plan(start, goal.cpu().numpy())
                if res.success:
                    return res
        return PlanResult(False, reason="no valid plans found")

    def run_exploration(
        self,
        rate: int = 10,
        manual_wait: bool = False,
        explore_iter: int = 3,
        try_to_plan_iter: int = 10,
        dry_run: bool = False,
        random_goals: bool = False,
        visualize: bool = False,
        task_goal: str = None,
        go_home_at_end: bool = False,
        show_goal: bool = False,
    ) -> Optional[Instance]:
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        self.current_state = "EXPLORE"
        self.robot.move_to_nav_posture()
        all_starts = []
        all_goals = []

        # Explore some number of times
        matches = []
        no_success_explore = True
        for i in range(explore_iter):
            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start, verbose=True)
            # if start is not valid move backwards a bit
            if not start_is_valid:
                print("Start not valid. back up a bit.")
                print(f"robot base pose: {self.robot.get_base_pose()}")
                self.robot.navigate_to([-0.1, 0, 0], relative=True)
                continue

            # Now actually plan to the frontier
            print("       Start:", start)
            self.print_found_classes(task_goal)
            res = self.plan_to_frontier(
                start=start, rate=rate, random_goals=random_goals, try_to_plan_iter=try_to_plan_iter
            )

            # if it succeeds, execute a trajectory to this position
            if res.success:
                no_success_explore = False
                print("Plan successful!")
                for i, pt in enumerate(res.trajectory):
                    print(i, pt.state)
                all_starts.append(start)
                all_goals.append(res.trajectory[-1].state)
                if visualize:
                    print("Showing goal location:")
                    robot_center = np.zeros(3)
                    robot_center[:2] = self.robot.get_base_pose()[:2]
                    self.voxel_map.show(
                        orig=robot_center,
                        xyt=res.trajectory[-1].state,
                        footprint=self.robot.get_robot_model().get_footprint(),
                        instances=self.semantic_sensor is not None,
                    )
                if not dry_run:
                    self.robot.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )
            else:
                # if it fails, try again or just quit
                if self._retry_on_fail:
                    print("Failed. Try again!")
                    continue
                else:
                    print("Failed. Quitting!")
                    break

            # Error handling
            if self.robot.last_motion_failed():
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("ROBOT IS STUCK! Move back!")
                print(f"robot base pose: {self.robot.get_base_pose()}")
                # Note that this is some random-walk code from habitat sim
                # This is a terrible idea, do not execute on a real robot
                # Not yet at least
                raise RuntimeError("Robot is stuck!")

                r = np.random.randint(3)
                if r == 0:
                    self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                elif r == 1:
                    self.robot.navigate_to([0, 0, np.pi / 4], relative=True, blocking=True)
                elif r == 2:
                    self.robot.navigate_to([0, 0, -np.pi / 4], relative=True, blocking=True)

            # Append latest observations
            self.update()
            # self.save_svm("", filename=f"debug_svm_{i:03d}.pkl")
            if visualize:
                # After doing everything - show where we will move to
                robot_center = np.zeros(3)
                robot_center[:2] = self.robot.get_base_pose()[:2]
                self.voxel_map.show(
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
                    break

        if go_home_at_end:
            self.current_state = "NAV_TO_HOME"
            # Finally - plan back to (0,0,0)
            print("Go back to (0, 0, 0) to finish...")
            start = self.robot.get_base_pose()
            goal = np.array([0, 0, 0])
            res = self.planner.plan(start, goal)
            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Full plan to home:")
                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)
                if not dry_run:
                    self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            else:
                print("WARNING: planning to home failed!")
        return matches

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
            self.robot.navigate_to(goal, blocking=False, timeout=30.0)
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
        self.voxel_map.reset()

    def save_instance_images(self, root: Union[Path, str] = ".", verbose: bool = False) -> None:
        """Save out instance images from the voxel map that we have collected while exploring."""

        if isinstance(root, str):
            root = Path(root)

        # Write out instance images
        for i, instance in enumerate(self.voxel_map.get_instances()):
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
        """Hae the robot say something out loud. This will send the text over to the robot from wherever the client is running."""
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

    def load_map(self, filename: str) -> None:
        """Load a map from a PKL file. Creates a new voxel map and loads the data from the file into this map. Then uses RANSAC to figure out where the current map and the loaded map overlap, computes a transform, and applies this transform to the loaded map to align it with the current map.

        Args:
            filename(str): the name of the file to load the map from
        """

        # Load the map from the file
        loaded_voxel_map = self._create_voxel_map(self.parameters)
        loaded_voxel_map.read_from_pickel(filename, perception=self.semantic_sensor)

        # Align the loaded map with the current map
        raise NotImplementedError("This function is not implemented yet.")

    def get_detections(self, **kwargs) -> List[Instance]:
        """Get the current detections."""
        instances = self.voxel_map.get_instances()

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
