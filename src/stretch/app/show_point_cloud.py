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
import numpy as np
import trimesh

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.utils.point_cloud import show_point_cloud


@click.command()
@click.option(
    "--robot_ip", default="", help="IP address of the robot (blank to use stored IP address)"
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
def main(
    robot_ip: str = "192.168.1.69",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    reset: bool = False,
):
    """Set up the robot and send it to home (0, 0, 0)."""
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    if reset:
        demo = RobotAgent(robot, parameters, None)
        demo.start(visualize_map_at_start=False, can_move=True)
        demo.move_closed_loop([0, 0, 0], max_time=60.0)
        robot.move_to_manip_posture()
    else:
        print("Starting")
        robot.start()

    servo = None
    obs = None
    while servo is None or obs is None:
        servo = robot.get_servo_observation()
        obs = robot.get_observation()

        if servo is not None:
            head_xyz = servo.get_xyz_in_world_frame(scaling=1e-3).reshape(-1, 3)
            ee_xyz = servo.get_ee_xyz_in_world_frame(scaling=1e-3).reshape(-1, 3)

            xyz = np.concatenate([head_xyz, ee_xyz], axis=0)
            rgb = (
                np.concatenate([servo.rgb.reshape(-1, 3), servo.ee_rgb.reshape(-1, 3)], axis=0)
                / 255
            )
            show_point_cloud(xyz, rgb, orig=np.zeros(3))
            break

        time.sleep(0.01)

    robot.stop()


if __name__ == "__main__":
    main()
