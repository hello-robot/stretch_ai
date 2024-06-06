import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Operation, Task

from .operations import (
    GoToNavOperation,
    GraspObjectOperation,
    NavigateToObjectOperation,
    PlaceObjectOperation,
    PreGraspObjectOperation,
    RotateInPlaceOperation,
    SearchForObjectOnFloorOperation,
    SearchForReceptacle,
)


class PickupManager:
    """Simple robot that will look around and pick up different objects"""

    def __init__(self, agent: RobotAgent) -> None:

        # Agent wraps high level functionality
        self.agent = agent

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

    def get_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan with loopbacks and recovery from failure"""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation("go to navigation mode", self)

        if add_rotate:
            # Spin in place to find objects.
            rotate_in_place = RotateInPlaceOperation(
                "Rotate in place", self, parent=go_to_navigation_mode
            )

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacle(
            "Search for a box",
            self,
            parent=rotate_in_place if add_rotate else go_to_navigation_mode,
            retry_on_failure=True,
        )

        # Try to expand the frontier and find an object; or just wander around for a while.
        search_for_object = SearchForObjectOnFloorOperation(
            "Search for toys on the floor", self, retry_on_failure=True
        )

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            "go to object",
            self,
            parent=search_for_object,
            on_cannot_start=search_for_object,
            to_receptacle=False,
        )

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_receptacle = NavigateToObjectOperation(
            "go to receptacle", self, on_cannot_start=search_for_receptacle, to_receptacle=True
        )

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        # If we cannot find the object, we should go back to the search_for_object operation.
        # To determine if we can start, we just check to see if there's a detectable object nearby.
        pregrasp_object = PreGraspObjectOperation(
            "prepare to grasp and make sure we can see the object",
            self,
            on_failure=None,
            on_cannot_start=go_to_object,
            retry_on_failure=True,
        )
        # If we cannot start, we should go back to the search_for_object operation.
        # To determine if we can start, we look at the end effector camera and see if there's anything detectable.
        grasp_object = GraspObjectOperation(
            "grasp the object",
            self,
            parent=pregrasp_object,
            on_failure=pregrasp_object,
            on_cannot_start=go_to_object,
        )
        place_object_on_receptacle = PlaceObjectOperation(
            "place object on receptacle", self, on_cannot_start=go_to_receptacle
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
