# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

from stretch.agent.operations import (
    GoToNavOperation,
    GraspObjectOperation,
    NavigateToObjectOperation,
    OpenLoopGraspObjectOperation,
    PreGraspObjectOperation,
    RotateInPlaceOperation,
    SearchForObjectOnFloorOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Operation, Task


class PickObjectTask:
    """Simple robot that will search for an object."""

    def __init__(
        self,
        agent: RobotAgent,
        target_object: Optional[str] = None,
        use_visual_servoing_for_grasp: bool = False,
        matching: str = "feature",
    ) -> None:
        # super().__init__(agent)
        self.agent = agent

        # Task information
        self.agent.target_object = target_object
        self.target_object = target_object

        assert matching in ["feature", "class"], f"Invalid instance matching method: {matching}"
        self.matching = matching

        # Sync these things
        self.robot = self.agent.robot
        self.voxel_map = self.agent.get_voxel_map()
        self.navigation_space = self.agent.space
        self.semantic_sensor = self.agent.semantic_sensor
        self.parameters = self.agent.parameters
        self.use_visual_servoing_for_grasp = use_visual_servoing_for_grasp
        self.instance_memory = self.agent.get_voxel_map().instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        self.current_object = None
        self.agent.reset_object_plans()

    def get_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan with loopbacks and recovery from failure. The robot will explore the environment, find objects, and pick them up

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.
            mode (str, optional): Type of task to create. Can be "one_shot" or "all". Defaults to "one_shot".

        Returns:
            Task: Executable task plan for the robot to pick up objects in the environment.
        """
        return self.get_one_shot_task(add_rotate=add_rotate, matching=self.matching)

    def get_all_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan that will pick up all objects in the environment. It starts by exploring the robot's immediate area, then will move around picking up all available objects.

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.

        Returns:
            Task: Executable task plan for the robot to pick up all objects in the environment.
        """
        raise NotImplementedError("This method is not yet implemented.")

    def get_one_shot_task(self, add_rotate: bool = False, matching: str = "feature") -> Task:
        """Create a task plan that will pick up a single object in the environment. It will explore until it finds a single object, and will then pick it up and place it in a receptacle."""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation(
            "go to navigation mode", self.agent, retry_on_failure=True
        )

        if add_rotate:
            # Spin in place to find objects.
            rotate_in_place = RotateInPlaceOperation(
                "rotate_in_place", self.agent, parent=go_to_navigation_mode
            )

        # Try to expand the frontier and find an object; or just wander around for a while.
        search_for_object = SearchForObjectOnFloorOperation(
            name=f"search_for_{self.target_object}_on_floor",
            agent=self.agent,
            retry_on_failure=True,
            match_method=matching,
            require_receptacle=False,
        )
        if self.agent.target_object is not None:
            # Overwrite the default object to search for
            search_for_object.set_target_object_class(self.agent.target_object)

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            name="go_to_object",
            agent=self.agent,
            parent=search_for_object,
            on_cannot_start=search_for_object,
            to_receptacle=False,
        )

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        # If we cannot find the object, we should go back to the search_for_object operation.
        # To determine if we can start, we just check to see if there's a detectable object nearby.
        pregrasp_object = PreGraspObjectOperation(
            name="prepare_to_grasp",
            agent=self.agent,
            on_failure=None,
            on_cannot_start=go_to_object,
            retry_on_failure=True,
        )

        # If we cannot start, we should go back to the search_for_object operation.
        # To determine if we can start, we look at the end effector camera and see if there's anything detectable.
        grasp_object: Operation = None
        if self.use_visual_servoing_for_grasp:
            grasp_object = GraspObjectOperation(
                f"grasp_the_{self.target_object}",
                self.agent,
                parent=pregrasp_object,
                on_failure=pregrasp_object,
                on_cannot_start=go_to_object,
                retry_on_failure=False,
            )
            grasp_object.set_target_object_class(self.agent.target_object)
            grasp_object.servo_to_grasp = True
            grasp_object.match_method = matching
        else:
            grasp_object = OpenLoopGraspObjectOperation(
                f"grasp_the_{self.target_object}",
                self.agent,
                parent=pregrasp_object,
                on_failure=pregrasp_object,
                on_cannot_start=go_to_object,
                retry_on_failure=False,
            )
            grasp_object.set_target_object_class(self.agent.target_object)
            grasp_object.match_method = matching

        task = Task()
        task.add_operation(go_to_navigation_mode)
        if add_rotate:
            task.add_operation(rotate_in_place)
        task.add_operation(search_for_object)
        task.add_operation(go_to_object)
        task.add_operation(pregrasp_object)
        task.add_operation(grasp_object)

        # Add success connections
        if add_rotate:
            task.connect_on_success(go_to_navigation_mode.name, rotate_in_place.name)
            task.connect_on_success(rotate_in_place.name, search_for_object.name)
        else:
            task.connect_on_success(go_to_navigation_mode.name, search_for_object.name)

        # Add success connections
        task.connect_on_success(search_for_object.name, go_to_object.name)
        task.connect_on_success(go_to_object.name, pregrasp_object.name)
        task.connect_on_success(pregrasp_object.name, grasp_object.name)

        # On failure try to search again
        task.connect_on_failure(pregrasp_object.name, search_for_object.name)
        task.connect_on_failure(grasp_object.name, search_for_object.name)

        # Add failures
        if add_rotate:
            # If we fail to find an object, rotate in place to find one.
            task.connect_on_failure(go_to_object.name, rotate_in_place.name)
        else:
            # If we fail to find an object, go back to the beginning and search again.
            task.connect_on_failure(go_to_object.name, search_for_object.name)

        # Terminate the task on successful grasp
        task.terminate_on_success(grasp_object.name)

        return task


if __name__ == "__main__":
    from stretch.agent.robot_agent import RobotAgent
    from stretch.agent.zmq_client import HomeRobotZmqClient

    robot = HomeRobotZmqClient()

    # Create a robot agent with instance memory
    agent = RobotAgent(robot, create_semantic_sensor=True)

    task = PickObjectTask(agent, target_object="stuffed leopard toy").get_task(add_rotate=False)
    task.run()
