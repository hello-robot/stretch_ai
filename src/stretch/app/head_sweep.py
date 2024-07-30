#!/usr/bin/env python
"""Sweep the head around and create a 3d map, then visualize it."""

import os
import sys

import click
import cv2
import numpy as np

from stretch.agent.zmq_client import HomeRobotZmqClient


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option(
    "--iterations", default=100, help="Number of iterations between rate histogram updates"
)
def main(
    robot_ip: str = "",
    local: bool = False,
    iterations: int = 500,
):

    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
    )

    observation = robot.get_observation()
    robot.move_to_nav_posture()
    # robot.move_to_manip_posture()

    # Wait and then...
    robot.head_to(head_pan=0, head_tilt=0, blocking=True)
    robot.head_to(head_pan=np.pi / 2, head_tilt=0, blocking=True)
    robot.head_to(head_pan=np.pi, head_tilt=0, blocking=True)
    robot.head_to(head_pan=0, head_tilt=0, blocking=True)

    robot.stop()


if __name__ == "__main__":
    main()
