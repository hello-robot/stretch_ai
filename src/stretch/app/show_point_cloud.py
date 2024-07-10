#!/usr/bin/env python3

import click

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.utils.point_cloud import show_point_cloud


@click.command()
@click.option("--robot_ip", default="192.168.1.69", help="IP address of the robot")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
def main(
    robot_ip: str = "192.168.1.69",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    reset: bool = False,
):
    """Set up the robot and send it to home (0, 0, 0)."""
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    demo = RobotAgent(robot, parameters, None)
    demo.start(visualize_map_at_start=False)
    if reset:
        demo.move_closed_loop([0, 0, 0], max_time=60.0)

    servo = None
    while servo is None:
        servo = robot.get_servo_observation()

        if servo is not None:
            breakpoint()

        time.sleep(0.01)

    robot.stop()


if __name__ == "__main__":
    main()
