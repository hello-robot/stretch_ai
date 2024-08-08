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

from stretch.agent.operations import (
    AvertGazeOperation,
    NodHeadOperation,
    ShakeHeadOperation,
    WaveOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.emote import EmoteManager
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters


def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "default_planner.yaml",
):
    # Create robot client
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )

    robot.move_to_nav_posture()

    # create robot agent
    demo = RobotAgent(robot, parameters=parameters)

    # create task manager
    manager = EmoteManager(demo)
    task = manager.get_task(NodHeadOperation("emote", manager))

    # run task
    task.run()

    task = manager.get_task(ShakeHeadOperation("emote", manager))

    task.run()

    task = manager.get_task(WaveOperation("emote", manager))

    task.run()

    task = manager.get_task(AvertGazeOperation("emote", manager))

    task.run()


if __name__ == "__main__":
    main()
