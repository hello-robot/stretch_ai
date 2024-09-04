#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import click
import numpy as np

from stretch.agent.operations import GraspObjectOperation, UpdateOperation
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.core.task import Task
from stretch.perception import create_semantic_sensor


def get_task(robot, demo, target_object):
    """Create a very simple task just to test visual servoing to grasp."""
    try:
        task = Task()
        update = UpdateOperation("update_scene", demo, retry_on_failure=True)
        grasp_object = GraspObjectOperation(
            "grasp_the_object",
            demo,
        )
        grasp_object.configure(
            show_object_to_grasp=True,
            servo_to_grasp=True,
            show_servo_gui=True,
            reset_observation=False,
            grasp_loose=(target_object == "cup"),
        )
        task.add_operation(update)
        task.add_operation(grasp_object)
    except Exception as e:
        print("Error in creating task: ", e)
        robot.stop()
        raise e
    return task


@click.command()
@click.option(
    "--robot_ip", default="", help="IP address of the robot (blank to use stored IP address)"
)
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--target_object", type=str, default="toy", help="Type of object to pick up and move")
@click.option(
    "--repeat_count", type=int, default=1, help="Number of times to repeat the grasp - for testing"
)
def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "toy",
    repeat_count: int = 1,
):
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    semantic_sensor = create_semantic_sensor(
        parameters,
        device_id=device_id,
        verbose=verbose,
        confidence_threshold=0.3,
    )

    # Start moving the robot around
    grasp_client = None

    # Agents wrap the robot high level planning interface for now
    demo = RobotAgent(robot, parameters, semantic_sensor, grasp_client=grasp_client)
    demo.start(visualize_map_at_start=show_intermediate_maps)

    targets = ["cup", "hand_towel", "screwdriver"]

    # for _ in range(repeat_count):
    for target_object in targets:
        if reset:
            robot.move_to_nav_posture()
            robot.navigate_to([0.0, 0.0, 0.0], blocking=True, timeout=30.0)
            # robot.arm_to([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], blocking=True)
            robot.arm_to([0.0, 0.78, 0.05, 0, -3 * np.pi / 8, 0], blocking=True)
            time.sleep(3.0)

        task = get_task(robot, demo, target_object)
        task.run()
        robot.open_gripper()

    if reset:
        robot.move_to_nav_posture()
        robot.navigate_to([0.0, 0.0, 0.0], blocking=True, timeout=30.0)

    robot.stop()


if __name__ == "__main__":
    main()
