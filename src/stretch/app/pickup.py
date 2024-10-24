#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.pickup import PickupTask
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option(
    "--target_object",
    type=str,
    default="toy",
    help="Type of object to pick up from the floor and move",
)
@click.option(
    "--receptacle",
    "--target_receptacle",
    type=str,
    default="box",
    help="Type of receptacle to place the object in",
)
@click.option(
    "--force-rotate",
    "--force_rotate",
    is_flag=True,
    help="Force the robot to rotate in place before doing anything.",
)
@click.option(
    "--match_method",
    type=click.Choice(["class", "feature"]),
    default="class",
    help="Method to match objects to pick up. Options: class, feature.",
    show_default=True,
)
@click.option(
    "--mode",
    default="one_shot",
    help="Mode of operation for the robot.",
    type=click.Choice(["one_shot", "all"]),
)
@click.option("--open_loop", is_flag=True, help="Use open loop grasping")
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    open_loop: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "shoe",
    receptacle: str = "box",
    force_rotate: bool = False,
    mode: str = "one_shot",
    match_method: str = "feature",
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    semantic_sensor = create_semantic_sensor(
        parameters=parameters,
        device_id=device_id,
        verbose=verbose,
    )

    # Agents wrap the robot high level planning interface for now
    agent = RobotAgent(robot, parameters, semantic_sensor)
    agent.start(visualize_map_at_start=show_intermediate_maps)
    if reset:
        agent.move_closed_loop([0, 0, 0], max_time=60.0)

    # After the robot has started...
    try:
        pickup_task = PickupTask(
            agent,
            target_object=target_object,
            target_receptacle=receptacle,
            matching=match_method,
            use_visual_servoing_for_grasp=not open_loop,
        )
        task = pickup_task.get_task(add_rotate=force_rotate, mode=mode)
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
