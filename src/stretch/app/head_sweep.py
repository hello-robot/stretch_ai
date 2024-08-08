#!/usr/bin/env python
# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

"""Sweep the head around and create a 3d map, then visualize it."""

import os
import sys
import time

import click
import cv2
import numpy as np

from stretch.agent import RobotAgent, RobotClient
from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option(
    "--run_semantic_segmentation", is_flag=True, help="Run semantic segmentation on EE rgb images"
)
@click.option("--show-open3d", is_flag=True, help="Show the open3d visualization")
@click.option("--device_id", type=int, default=0, help="Device ID for camera")
@click.option("--verbose", is_flag=True, help="Print debug information from perception")
def main(
    robot_ip: str = "",
    local: bool = False,
    run_semantic_segmentation: bool = False,
    show_open3d: bool = False,
    device_id: int = 0,
    verbose: bool = False,
):

    # Create robot
    robot = RobotClient(robot_ip=robot_ip, use_remote_computer=(not local))
    if run_semantic_segmentation:
        _, semantic_sensor = create_semantic_sensor(
            device_id=device_id,
            verbose=verbose,
            category_map_file=robot.parameters["open_vocab_category_map_file"],
        )
    else:
        semantic_sensor = None
    agent = RobotAgent(robot, robot.parameters, semantic_sensor)

    observation = robot.get_observation()
    robot.move_to_nav_posture()

    # Wait and then...
    robot.head_to(head_pan=0, head_tilt=0, blocking=True)
    agent.update()
    robot.head_to(head_pan=-np.pi / 2, head_tilt=0, blocking=True)
    agent.update()
    robot.head_to(head_pan=-np.pi, head_tilt=0, blocking=True)
    agent.update()
    robot.head_to(head_pan=0, head_tilt=0, blocking=True)
    agent.update()
    robot.head_to(head_pan=0, head_tilt=-np.pi / 2, blocking=True)
    agent.update()
    robot.head_to(head_pan=0, head_tilt=0, blocking=True)
    agent.update()

    if show_open3d:
        agent.show_map()
        print("Done.")
    else:
        input("Press Enter when done...")
    robot.stop()


if __name__ == "__main__":
    main()
