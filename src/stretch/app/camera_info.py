#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

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
    )
    robot.start()
    try:
        while True:
            servo = robot.get_servo_observation()
            obs = robot.get_observation()

            if servo is None:
                continue
            if obs is None:
                continue

            print("---------------------- Camera Info ----------------------")
            print(
                f"Servo Head RGB shape: {servo.rgb.shape} Servo Head Depth shape: {servo.depth.shape}"
            )
            print(
                f"Servo EE RGB shape: {servo.ee_rgb.shape} Servo EE Depth shape: {servo.ee_depth.shape}"
            )
            print(
                f"Observation RGB shape: {obs.rgb.shape} Observation Depth shape: {obs.depth.shape}"
            )
            break

    except KeyboardInterrupt:
        pass
    robot.stop()


if __name__ == "__main__":
    main()
