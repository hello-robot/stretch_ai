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

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--open", is_flag=True, help="Open the gripper")
@click.option("--close", is_flag=True, help="Close the gripper")
@click.option("--value", default=0.0, help="Value to set the gripper to")
@click.option("--blocking", is_flag=True, help="Block until the gripper is done")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    open: bool = False,
    close: bool = False,
    blocking: bool = False,
    value: float = 0.0,
):
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    print("Starting")
    robot.start()
    if open:
        robot.open_gripper(blocking=blocking)
    if close:
        robot.close_gripper(blocking=blocking)
    if not open and not close:
        robot.gripper_to(value, blocking=blocking)
    print("Done.")
    robot.stop()


if __name__ == "__main__":
    main()
