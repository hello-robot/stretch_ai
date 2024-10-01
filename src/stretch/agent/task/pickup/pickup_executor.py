# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple

import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.core import AbstractRobotClient


class PickupExecutor:
    """This class parses commands from the pickup llm bot and sends them to the robot."""

    def __init__(self, robot: AbstractRobotClient, agent: RobotAgent):
        self.robot = robot
        self.agent = agent

    def __call__(self, response: List[Tuple[str, str]]) -> None:
        """Execute the list of commands."""
        for i, (command, args) in enumerate(response):
            if command == "say":
                logger.info(f"{i} {command} {args}")
                self.agent.robot_say(args)
            elif command == "pickup":
                self.agent.pickup(args)
            elif command == "place":
                self.agent.place(args)
            elif command == "wave":
                self.agent.wave()
            elif command == "go_home":
                self.agent.go_home()
            elif command == "explore":
                self.agent.explore()
            elif command == "nod_head":
                self.agent.nod_head()
            elif command == "shake_head":
                self.agent.shake_head()
            elif command == "avert_gaze":
                self.agent.avert_gaze()
            elif command == "end":
                break
            else:
                # logger.error(f"Unknown command: {command}")
                pass
