# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

from typing import Optional, Union

import numpy as np

import stretch.utils.logger as logger
from stretch.agent.base import TaskManager
from stretch.agent.operations import (
    GoToNavOperation,
    GraspObjectOperation,
    NavigateToObjectOperation,
    PlaceObjectOperation,
    PreGraspObjectOperation,
    RotateInPlaceOperation,
    SearchForObjectOnFloorOperation,
    SearchForReceptacleOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task
from stretch.mapping.instance import Instance


class PickupManager(TaskManager):
    """Simple robot that will look around and pick up different objects"""

    def __init__(
        self,
        agent: RobotAgent,
        target_object: Optional[str] = None,
        use_visual_servoing_for_grasp: bool = True,
    ) -> None:
        super().__init__(agent)

        # Task information
        self.target_object = target_object
        self.use_visual_servoing_for_grasp = use_visual_servoing_for_grasp

        # Sync these things
        self.robot = agent.robot
        self.voxel_map = agent.voxel_map
        self.navigation_space = agent.space
        self.semantic_sensor = agent.semantic_sensor
        self.parameters = agent.parameters
        self.instance_memory = agent.voxel_map.instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        self.current_object = None
        self.current_receptacle = None
        self.reset_object_plans()

    def get_task(self, add_rotate: bool = False, mode: str = "one_shot") -> Task:
        """Create a task plan with loopbacks and recovery from failure. The robot will explore the environment, find objects, and pick them up

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.
            mode (str, optional): Type of task to create. Can be "one_shot" or "all". Defaults to "one_shot".

        Returns:
            Task: Executable task plan for the robot to pick up objects in the environment.
        """

        if mode == "one_shot":
            return self.get_one_shot_task(add_rotate=add_rotate)
        elif mode == "all":
            if not add_rotate:
                logger.warning(
                    "When performing pickup task in 'all' mode, we must add a rotate operation to explore the robot's area to identify multiple object instances."
                )
            return self.get_all_task(add_rotate=True)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_all_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan that will pick up all objects in the environment. It starts by exploring the robot's immediate area, then will move around picking up all available objects.

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.

        Returns:
            Task: Executable task plan for the robot to pick up all objects in the environment.
        """

    def get_one_shot_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan that will pick up a single object in the environment. It will explore until it finds a single object, and will then pick it up and place it in a receptacle."""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation(
            "go to navigation mode", self, retry_on_failure=True
        )

        if add_rotate:
            # Spin in place to find objects.
            rotate_in_place = RotateInPlaceOperation(
                "rotate_in_place", self, parent=go_to_navigation_mode
            )

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacleOperation(
            "search_for_box",
            self,
            parent=rotate_in_place if add_rotate else go_to_navigation_mode,
            retry_on_failure=True,
        )

        # Try to expand the frontier and find an object; or just wander around for a while.
        search_for_object = SearchForObjectOnFloorOperation(
            "search_for_objects_on_floor", self, retry_on_failure=True
        )
        if self.target_object is not None:
            # Overwrite the default object to search for
            search_for_object.set_target_object_class(self.target_object)

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            "go_to_object",
            self,
            parent=search_for_object,
            on_cannot_start=search_for_object,
            to_receptacle=False,
        )

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_receptacle = NavigateToObjectOperation(
            "go_to_receptacle", self, on_cannot_start=search_for_receptacle, to_receptacle=True
        )

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        # If we cannot find the object, we should go back to the search_for_object operation.
        # To determine if we can start, we just check to see if there's a detectable object nearby.
        pregrasp_object = PreGraspObjectOperation(
            "prepare_to_grasp",
            self,
            on_failure=None,
            on_cannot_start=go_to_object,
            retry_on_failure=True,
        )
        # If we cannot start, we should go back to the search_for_object operation.
        # To determine if we can start, we look at the end effector camera and see if there's anything detectable.
        grasp_object = GraspObjectOperation(
            "grasp_the_object",
            self,
            parent=pregrasp_object,
            on_failure=pregrasp_object,
            on_cannot_start=go_to_object,
            retry_on_failure=False,
        )
        grasp_object.servo_to_grasp = self.use_visual_servoing_for_grasp
        place_object_on_receptacle = PlaceObjectOperation(
            "place_object_on_receptacle", self, on_cannot_start=go_to_receptacle
        )

        task = Task()
        task.add_operation(go_to_navigation_mode)
        if add_rotate:
            task.add_operation(rotate_in_place)
        task.add_operation(search_for_receptacle)
        task.add_operation(search_for_object)
        task.add_operation(go_to_object)
        task.add_operation(pregrasp_object)
        task.add_operation(grasp_object)
        task.add_operation(go_to_receptacle)
        task.add_operation(place_object_on_receptacle)

        task.connect_on_success(go_to_navigation_mode.name, search_for_receptacle.name)
        task.connect_on_success(search_for_receptacle.name, search_for_object.name)
        task.connect_on_success(search_for_object.name, go_to_object.name)
        task.connect_on_success(go_to_object.name, pregrasp_object.name)
        task.connect_on_success(pregrasp_object.name, grasp_object.name)
        task.connect_on_success(grasp_object.name, go_to_receptacle.name)
        task.connect_on_success(go_to_receptacle.name, place_object_on_receptacle.name)

        task.connect_on_success(search_for_receptacle.name, search_for_object.name)

        task.connect_on_cannot_start(go_to_object.name, search_for_object.name)
        # task.connect_on_cannot_start(go_to_receptacle.name, search_for_receptacle.name)

        return task
