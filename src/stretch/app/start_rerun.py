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
import timeit

import click

from stretch.agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--test-svm", is_flag=True, help="Set if we are testing the planner")
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "default_planner.yaml",
    test_svm: bool = False,
):
    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        enable_rerun_server=True,
    )
    robot.start()
    if test_svm:
        parameters = get_parameters(parameter_file)
        agent = RobotAgent(robot, parameters)
        agent.update()
        t0 = timeit.default_timer()
        agent.update_rerun()
        t1 = timeit.default_timer()
        print(f"Time to update rerun: {t1 - t0}")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    robot.stop()


if __name__ == "__main__":
    main()
