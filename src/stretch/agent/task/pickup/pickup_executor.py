# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.agent.task.pickup.pickup_task import PickupTask
from stretch.core import AbstractRobotClient
from stretch.utils.logger import Logger

logger = Logger(__name__)
# Default to hiding info messages
# logger.hide_info()


class PickupExecutor:
    """This class parses commands from the pickup llm bot and sends them to the robot."""

    _pickup_task_mode = "one_shot"

    def __init__(
        self,
        robot: AbstractRobotClient,
        agent: RobotAgent,
        match_method: str = "feature",
        open_loop: bool = False,
        dry_run: bool = False,
        available_actions: List[str] = None,
    ) -> None:
        """Initialize the executor.

        Args:
            robot: The robot client.
            agent: The robot agent.
            dry_run: If true, don't actually execute the commands.
        """
        self.robot = robot
        self.agent = agent
        self.available_actions = available_actions

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")
        if not isinstance(self.agent, RobotAgent):
            raise TypeError(f"Expected RobotAgent, got {type(self.agent)}")

        self.dry_run = dry_run
        self.emote_task = EmoteTask(self.agent)

        # Configuration
        self._match_method = match_method
        self._open_loop = open_loop

    def _pickup(self, target_object: str, target_receptacle: str) -> None:

        # After the robot has started...
        try:
            pickup_task = PickupTask(
                self.agent,
                target_object=target_object,
                target_receptacle=target_receptacle,
                matching=self._match_method,
                use_visual_servoing_for_grasp=not self._open_loop,
            )
            task = pickup_task.get_task(add_rotate=True, mode=self._pickup_task_mode)
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def __call__(self, response: List[Tuple[str, str]]) -> bool:
        """Execute the list of commands given by the LLM bot.

        Args:
            response: A list of tuples, where the first element is the command and the second is the argument.

        Returns:
            True if we should keep going, False if we should stop.
        """
        i = 0

        if response is None or len(response) == 0:
            logger.error("No commands to execute!")
            self.agent.robot_say("I'm sorry, I didn't understand that.")
            return True

        # Loop over every command we have been given
        # Pull out pickup and place as a single arg if they are in a row
        # Else, execute things as they come
        while i < len(response):
            command, args = response[i]
            logger.info(f"Command: {i} {command} {args}")
            if command == "say":
                # Use TTS to say the text
                logger.info(f"Saying: {args}")
                self.agent.robot_say(args)
            elif command == "pickup":
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args
                i += 1
                next_command, next_args = response[i]
                if next_command != "place":
                    i -= 1
                    logger.error("Pickup without place! Doing nothing.")
                else:
                    logger.info(f"{i} {next_command} {next_args}")
                    logger.info(f"[Pickup task] Place: {next_args}")
                target_receptacle = next_args
                self._pickup(target_object, target_receptacle)
            elif command == "place":
                logger.error("Place without pickup! Doing nothing.")
            elif command == "wave":
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()
            elif command == "go_home":
                self.agent.go_home()
            elif command == "explore":
                self.agent.explore()
            elif command == "nod_head":
                self.emote_task.get_task("nod_head").run()
            elif command == "shake_head":
                self.emote_task.get_task("shake_head").run()
            elif command == "avert_gaze":
                self.emote_task.get_task("avert_gaze").run()
            elif command == "quit":
                logger.info("[Pickup task] Quitting.")
                self.robot.stop()
                return False
            elif command == "end":
                logger.info("[Pickup task] Ending.")
                break
            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1
        # If we did not explicitly receive a quit command, we are not yet done.
        return True
