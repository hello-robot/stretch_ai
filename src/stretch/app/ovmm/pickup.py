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

    def can_start(self) -> bool:
        return self.manager.found_receptacle()

    def run(self) -> None:
        print("Find a reachable object on the floor.")
        self._successful = False

        # Check to see if we have an object on the floor worth finding

        self._successful = True

    def was_successful(self) -> bool:
        return self._successful


class GraspObjectOperation(Operation):
    pass


class ResetArmOperation(Operation):
    def __init__(self, name, manager, **kwargs):
        super().__init__(name, **kwargs)
        self.manager = manager
        self.robot = manager.robot
        self.parameters = manager.parameters


class PickupManager:
    """Simple robot that will look around and pick up different objects"""

    def __init__(self, robot: RobotAgent, parameters: Parameters) -> None:
        self.robot = robot
        self.parameters = parameters
        self.found_receptacle = False
        self.receptacle = None

    def get_task(self):
        """Create a task"""
        task = Task()

        go_to_navigation_mode = GoToNavOperation("go to navigation mode", self)
        rotate_in_place = RotateInPlaceOperation(self, parent=go_to_navigation_mode)
        search_for_receptacle = SearchForReceptacle(self, parent=rotate_in_place)
        search_for_object = SearchForObjectOnFloorOperation(self, parent=search_for_receptacle)

        # These two are supposed to just move the object around
        manipulation_mode = ResetArmOperation(
            "go to manipulation mode", self, parent=search_for_object
        )
        reset_arm = ResetArmOperation("reset arm to retry", self, on_success=grasp_object)

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        grasp_object = GraspObjectOperation(
            self, parent=manipulation_mode, on_failure=reset_arm, on_cannot_start=search_for_object
        )

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
