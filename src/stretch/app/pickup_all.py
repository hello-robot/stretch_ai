#!/usr/bin/env python3

import click
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.pickup import PickupManager
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.core.task import Operation, Task
from stretch.mapping.voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace
from stretch.perception import create_semantic_sensor, get_encoder


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--target_object", type=str, default="toy", help="Type of object to pick up and move")
@click.option(
    "--force-rotate",
    "--force_rotate",
    is_flag=True,
    help="Force the robot to rotate in place before doing anything.",
)
@click.option(
    "--destination",
    type=str,
    default="box",
    help="Where to put the objects once you have found them",
)
@click.option(
    "--mode",
    default="one_shot",
    help="Mode of operation for the robot.",
    type=click.Choice(["one_shot", "all"]),
)
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "shoe",
    destination: str = "box",
    force_rotate: bool = False,
    mode: str = "one_shot",
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    _, semantic_sensor = create_semantic_sensor(
        device_id=device_id,
        verbose=verbose,
        category_map_file=parameters["open_vocab_category_map_file"],
    )

    # Start moving the robot around
    grasp_client = None

    # Agents wrap the robot high level planning interface for now
    agent = RobotAgent(robot, parameters, semantic_sensor, grasp_client=grasp_client)
    agent.start(visualize_map_at_start=show_intermediate_maps)
    if reset:
        agent.move_closed_loop([0, 0, 0], max_time=60.0)

    # After the robot has started...
    try:
        manager = PickupManager(agent, target_object=target_object, destination=destination)
        task = manager.get_task(add_rotate=force_rotate, mode=mode)
    except Exception as e:
        print(f"Error creating task: {e}")
        robot.stop()
        raise e

    task.run()

    if reset:
        # Send the robot home at the end!
        agent.go_home()

    # At the end, disable everything
    robot.stop()


if __name__ == "__main__":
    main()
