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

import click

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
):
    """Set up the robot and send it to home (0, 0, 0)."""
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    demo = RobotAgent(robot, parameters, None)
    demo.start(visualize_map_at_start=False)
    demo.move_closed_loop([0, 0, 0], max_time=60.0)
    robot.stop()


if __name__ == "__main__":
    main()
