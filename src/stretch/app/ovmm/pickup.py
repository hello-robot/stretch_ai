#!/usr/bin/env python3

import click

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, get_parameters
from stretch.core.task import Operation, Task


class RotateInPlaceOperation(Operation):
    """Rotate the robot in place"""

    def __init__(self, manager, **kwargs):
        super().__init__("Rotate in place", **kwargs)
        self.robot = manager.robot
        self.parameters = manager.parameters
        self.manager = manager

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        print(f"Running {self.name}")
        self._successful = False
        self.robot.rotate_in_place(
            steps=self.parameters["in_place_rotation_steps"],
            visualize=False,
        )
        self._successful = True

    def was_successful(self) -> bool:
        return self._successful


class SearchForReceptacle(Operation):
    """Find a place to put the objects we find on the floor"""

    def __init__(self, manager, **kwargs):
        super().__init__("Search for receptacle", **kwargs)
        self.manager = manager
        self.robot = self.manager.robot
        self.parameters = self.manager.parameters

    def can_start(self) -> bool:
        return True


class SearchForObjectOnFloorOperation(Operation):
    """Search for an object on the floor"""

    def __init__(self, manager, **kwargs):
        super().__init__("Search for object", **kwargs)
        self.robot = robot
        self.parameters = parameters

        # Overload failure to just retry this one
        if self.on_failure is not None:
            raise RuntimeError(
                "Cannot have on_failure set for SearchForObjectOnFloorOperation - it will just retry itself."
            )
        self.on_failure = self

    def can_start(self) -> bool:
        return self.manager.found_receptacle()

    def run(self) -> None:
        print("Find a reachable object on the floor.")
        self._successful = False

        # Check to see if we have an object on the floor worth finding

        # Check to see if there is a visitable frontier

        # If no visitable frontier, pick a random point nearby and just wander around

        self._successful = True

    def was_successful(self) -> bool:
        return self._successful and self.manager.found_receptacle()


class PreGraspObjectOperation(Operation):
    """Move the robot to a position looking at the object using the navigation/manipulation camera."""

    pass


class GraspObjectOperation(Operation):
    """Move the robot to grasp, using the end effector camera."""

    pass


class ResetArmOperation(Operation):
    """Send the arm back to home"""

    def __init__(self, name, manager, **kwargs):
        super().__init__(name, **kwargs)
        self.manager = manager
        self.robot = manager.robot
        self.parameters = manager.parameters

    def can_start(self) -> bool:
        """This one has no special requirements"""
        return True


class PickupManager:
    """Simple robot that will look around and pick up different objects"""

    def __init__(self, robot: RobotAgent, parameters: Parameters) -> None:
        self.robot = robot
        self.parameters = parameters
        self.found_receptacle = False
        self.receptacle = None

    def get_task(self):
        """Create a task"""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation("go to navigation mode", self)

        # Spin in place to find objects.
        rotate_in_place = RotateInPlaceOperation(self, parent=go_to_navigation_mode)

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacle(self, parent=rotate_in_place)

        # Try to expand the frontier and find an object; or just wander around for a while.
        search_for_object = SearchForObjectOnFloorOperation(self, parent=search_for_receptacle)

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = GoToObjectOperation(
            "go to object", self, parent=search_for_object, on_cannot_start=search_for_object
        )

        # These two are supposed to just move the object around
        manipulation_mode = ResetArmOperation(
            "go to manipulation mode", self, parent=search_for_object
        )
        reset_arm = ResetArmOperation("reset arm to retry", self, on_success=grasp_object)

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        # If we cannot find the object, we should go back to the search_for_object operation.
        # To determine if we can start, we just check to see if there's a detectable object nearby.
        pregrasp_object = PreGraspObjectOperation(
            self, parent=manipulation_mode, on_failure=reset_arm, on_cannot_start=go_to_object
        )
        # If we cannot start, we should go back to the search_for_object operation.
        # To determine if we can start, we look at the end effector camera and see if there's anything detectable.
        grasp_object = GraspObjectOperation(
            self, parent=pregrasp_object, on_failure=reset_arm, on_cannot_start=go_to_object
        )

        task.add_operation(go_to_navigation_mode)
        task.add_operation(rotate_in_place)
        task.add_operation(search_for_receptacle)
        task.add_operation(search_for_object)
        task.add_operation(go_to_object)
        task.add_operation(manipulation_mode)
        task.add_operation(pregrasp_object)
        task.add_operation(grasp_object)

        return task


@click.command()
@click.option("--robot_ip", default="192.168.1.15", help="IP address of the robot")
@click.option("--recv_port", default=4401, help="Port to receive messages from the robot")
@click.option("--send_port", default=4402, help="Port to send messages to the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option(
    "--parameter_file", default="config/default_planner.yaml", help="Path to parameter file"
)
def main(
    robot_ip: str = "192.168.1.15",
    recv_port: int = 4401,
    send_port: int = 4402,
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    # Start moving the robot around
    demo = RobotAgent(robot, parameters, semantic_sensor, grasp_client=grasp_client)
    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)

    # After the robot has started...
    agent = PickupManager(robot, parameters)


if __name__ == "__main__":
    main()
