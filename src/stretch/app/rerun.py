# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
# 
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


#!/usr/bin/env python3

import time

import click
import cv2
import numpy as np

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.motion import HelloStretchIdx


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
):
    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        enable_rerun_server=True,
    )
    robot.start()
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    robot.stop()


if __name__ == "__main__":
    main()
