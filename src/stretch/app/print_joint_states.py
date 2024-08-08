#!/usr/bin/env python3

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
@click.option("-j", "--joint", default="", help="Joint to print")
def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    joint: str = "",
):
    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        enable_rerun_server=False,
    )
    robot.start()
    try:
        while True:
            joint_state = robot.get_joint_positions()
            if joint_state is None:
                continue
            if len(joint) > 0:
                print(f"{joint}: {joint_state[HelloStretchIdx.get_idx(joint.lower())]}")
            else:
                print(
                    f"Arm: {joint_state[HelloStretchIdx.ARM]}, Lift: {joint_state[HelloStretchIdx.LIFT]}, Gripper: {joint_state[HelloStretchIdx.GRIPPER]}"
                )
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    robot.stop()


if __name__ == "__main__":
    main()
