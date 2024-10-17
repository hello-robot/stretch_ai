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

from stretch.agent.operations import GraspObjectOperation, UpdateOperation
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.core.task import Task
from stretch.perception import create_semantic_sensor


def get_task(robot, demo, target_object):
    """Create a very simple task just to test visual servoing to grasp."""
    print("[GRASP OBJECT APP] Target object is set to", target_object)
    try:
        task = Task()
        update = UpdateOperation("update_scene", demo, retry_on_failure=True)
        update.configure(
            move_head=False,
            target_object=target_object,
            show_map_so_far=False,
            clear_voxel_map=True,
            show_instances_detected=False,
            match_method="feature",
            arm_height=0.6,
        )
        grasp_object = GraspObjectOperation(
            "grasp_the_object",
            demo,
        )
        grasp_object.configure(
            target_object=target_object,
            show_object_to_grasp=True,
            servo_to_grasp=True,
            show_servo_gui=True,
            reset_observation=False,
            grasp_loose=(target_object == "cup"),
            match_method="feature",
        )
        grasp_object.set_target_object_class(target_object)
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

    # Agents wrap the robot high level planning interface for now
    demo = RobotAgent(robot, parameters, semantic_sensor)
    demo.start(visualize_map_at_start=show_intermediate_maps)

    task = get_task(robot, demo, target_object)
    task.run()
    robot.open_gripper()

    if reset:
        robot.move_to_nav_posture()
        robot.navigate_to([0.0, 0.0, 0.0], blocking=True, timeout=30.0)

    robot.stop()


if __name__ == "__main__":
    main()
