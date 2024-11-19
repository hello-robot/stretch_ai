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
    NavigateToObjectOperation,
    PlaceObjectOperation,
    RotateInPlaceOperation,
    SearchForReceptacleOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task


class PlaceOnReceptacleTask:
    """Simple robot that will look around and pick up different objects"""

    def __init__(
        self,
        agent: RobotAgent,
        target_receptacle: Optional[str] = None,
        use_visual_servoing_for_grasp: bool = True,
        matching: str = "feature",
    ) -> None:
        # super().__init__(agent)
        self.agent = agent

        # Task information
        self.agent.target_receptacle = target_receptacle
        self.target_receptacle = target_receptacle

        assert matching in ["feature", "class"], f"Invalid instance matching method: {matching}"
        self.matching = matching

        # Sync these things
        self.robot = self.agent.robot
        self.voxel_map = self.agent.get_voxel_map()
        self.navigation_space = self.agent.space
        self.semantic_sensor = self.agent.semantic_sensor
        self.parameters = self.agent.parameters
        self.instance_memory = self.agent.get_voxel_map().instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        self.current_receptacle = None
        self.agent.reset_object_plans()

    def get_task(self, add_rotate: bool = False, mode: str = "one_shot") -> Task:
        """Create a task plan with loopbacks and recovery from failure. The robot will explore the environment, find objects, and pick them up

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.
            mode (str, optional): Type of task to create. Can be "one_shot" or "all". Defaults to "one_shot".

        Returns:
            Task: Executable task plan for the robot to pick up objects in the environment.
        """

        return self.get_one_shot_task(add_rotate=add_rotate, matching=self.matching)

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

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacleOperation(
            name=f"search_for_{self.target_receptacle}",
            agent=self.agent,
            parent=rotate_in_place if add_rotate else go_to_navigation_mode,
            retry_on_failure=True,
            match_method=matching,
        )

        search_for_receptacle.set_target_object_class(self.target_receptacle)

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_receptacle = NavigateToObjectOperation(
            name="go_to_receptacle",
            agent=self.agent,
            on_cannot_start=search_for_receptacle,
            to_receptacle=True,
        )

        place_object_on_receptacle = PlaceObjectOperation(
            name="place_object_on_receptacle",
            agent=self.agent,
            on_cannot_start=go_to_receptacle,
            require_object=False,
        )

        task = Task()
        task.add_operation(go_to_navigation_mode)
        if add_rotate:
            task.add_operation(rotate_in_place)
        task.add_operation(search_for_receptacle)
        task.add_operation(go_to_receptacle)
        task.add_operation(place_object_on_receptacle)

        task.connect_on_success(go_to_navigation_mode.name, search_for_receptacle.name)
        task.connect_on_success(search_for_receptacle.name, go_to_receptacle.name)
        task.connect_on_success(go_to_receptacle.name, place_object_on_receptacle.name)

        # Terminate on a successful place
        task.terminate_on_success(place_object_on_receptacle.name)

        return task


if __name__ == "__main__":
    from stretch.agent.robot_agent import RobotAgent
    from stretch.agent.zmq_client import HomeRobotZmqClient

    robot = HomeRobotZmqClient()
    # Create a robot agent with instance memory
    agent = RobotAgent(robot, create_semantic_sensor=True)

    task = PlaceOnReceptacleTask(agent, target_receptacle="cardboard box").get_task(
        add_rotate=False
    )
    task.run()
