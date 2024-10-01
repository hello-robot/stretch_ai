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
        i = 0

        # Loop over every command we have been given
        # Pull out pickup and place as a single arg if they are in a row
        # Else, execute things as they come
        while i < len(response):
            command, args = response[i]
            logger.info(f"{i} {command} {args}")
            if command == "say":
                self.agent.robot_say(args)
            elif command == "pickup":
                target_object = args
                i += 1
                next_command, next_args = response[i]
                if next_command != "place":
                    i -= 1
                    logger.error("Pickup without place! Doing nothing.")
                else:
                    logger.info(f"{i} {command} {args}")
                breakpoint()
            elif command == "place":
                logger.error("Place without pickup! Doing nothing.")
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
                logger.error(f"Skipping unknown command: {command}")
