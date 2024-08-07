# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
# 
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


#!/usr/bin/env python3

import click

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, get_parameters


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--open", is_flag=True, help="Open the gripper")
@click.option("--close", is_flag=True, help="Close the gripper")
@click.option("--blocking", is_flag=True, help="Block until the gripper is done")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    open: bool = False,
    close: bool = False,
    blocking: bool = False,
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
    robot.switch_to_manipulation_mode()
    if open:
        robot.open_gripper(blocking=blocking)
    if close:
        robot.close_gripper(blocking=blocking)
    print("Done.")
    robot.stop()


if __name__ == "__main__":
    main()
