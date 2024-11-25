# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple

from stretch.agent.task.emote import EmoteTask
from stretch.core import AbstractRobotClient
from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.utils.logger import Logger

# Mapping and perception
from stretch.core.parameters import get_parameters

logger = Logger(__name__)


class DynamemTaskExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient,
        match_method: str = "feature",
    ) -> None:
        """Initialize the executor.

        Args:
            robot: The robot client.
            agent: The robot agent.
            dry_run: If true, don't actually execute the commands.
        """
        self.robot = robot

        print("- Load parameters")
        parameters = get_parameters("dynav_config.yaml")
        self.parameters = parameters

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")

        # Configuration
        self._match_method = match_method

        # Create semantic sensor if visual servoing is enabled
        print("- Create semantic sensor if visual servoing is enabled")
        if visual_servo:
            semantic_sensor = create_semantic_sensor(
                parameters=parameters,
                device_id=device_id,
                verbose=False,
            )
        else:
            parameters["encoder"] = None
            semantic_sensor = None

        print("- Start robot agent with data collection")
        self.agent = RobotAgent(robot, parameters, semantic_sensor)
        self.agent.start()

        # Task stuff
        self.emote_task = EmoteTask(self.agent)


    def _pickup(self, target_object: str) -> None:
        """Pick up an object.

        Args:
            target_object: The object to pick up.
        """
        raise NotImplementedError

    def _place(self, target_receptacle: str) -> None:
        """Place an object.

        Args:
            target_receptacle: The receptacle to place the object in.
        """

        raise NotImplementedError

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

        logger.info("Resetting agent...")
        self.agent.reset()

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
                next_command, next_args = response[i]
                self._pickup(target_object)
            elif command == "place":
                logger.warning(
                    "Place without pickup! Try giving a full pick-and-place instruction."
                )
                self._place(args)
            elif command == "wave":
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()
            elif command == "go_home":
                if self.agent.get_voxel_map().is_empty():
                    logger.warning("No map data available. Cannot go home.")
                else:
                    self.agent.go_home()
            elif command == "explore":
                self.agent.explore()
            elif command == "find":
                self._find(args)
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
