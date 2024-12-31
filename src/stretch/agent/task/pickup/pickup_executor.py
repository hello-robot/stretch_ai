# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
from typing import List, Tuple

from PIL import Image

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.agent.task.pickup.find_task import FindObjectTask
from stretch.agent.task.pickup.hand_over_task import HandOverTask
from stretch.agent.task.pickup.pick_task import PickObjectTask
from stretch.agent.task.pickup.pickup_task import PickupTask
from stretch.agent.task.pickup.place_task import PlaceOnReceptacleTask
from stretch.core import AbstractRobotClient
from stretch.utils.image import numpy_image_to_bytes
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
        discord_bot=None,
    ) -> None:
        """Initialize the executor.

        Args:
            robot: The robot client.
            agent: The robot agent.
            dry_run: If true, don't actually execute the commands.
            available_actions: A list of available actions.
        """
        self.robot = robot
        self.agent = agent
        self.available_actions = available_actions

        # Optional discord integration for chatting with the robot
        self.discord_bot = discord_bot

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
        """Create a task to pick up the object and execute it.

        Args:
            target_object: The object to pick up.
            target_receptacle: The receptacle to place the object in.
        """

        if target_receptacle is None or len(target_receptacle) == 0:
            self._pick_only(target_object)
            return

        logger.alert(f"[Pickup task] Pickup: {target_object} Place: {target_receptacle}")

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

    def _take_picture(self, channel=None) -> None:
        """Take a picture with the head camera. Optionally send it to Discord."""

        obs = self.robot.get_observation()
        if channel is None:
            # Just save it to the disk
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel, message="Head camera:", content=numpy_image_to_bytes(obs.rgb)
            )

    def _take_ee_picture(self, channel=None) -> None:
        """Take a picture of the end effector."""

        obs = self.robot.get_servo_observation()
        if channel is None:
            # Just save it to the disk
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.ee_rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel,
                message="End effector camera:",
                content=numpy_image_to_bytes(obs.ee_rgb),
            )

    def _pick_only(self, target_object: str) -> None:
        """Create a task to pick up the object and execute it.

        Args:
            target_object: The object to pick up.
        """

        logger.alert(f"[Pickup task] Pickup: {target_object}")

        # After the robot has started...
        try:
            pickup_task = PickObjectTask(
                self.agent,
                target_object=target_object,
                matching=self._match_method,
                use_visual_servoing_for_grasp=not self._open_loop,
            )
            task = pickup_task.get_task(add_rotate=True)
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def _place(self, target_receptacle: str) -> None:
        """Create a task to place the object and execute it.

        Args:
            target_receptacle: The receptacle to place the object in.
        """
        logger.alert(f"[Pickup task] Place: {target_receptacle}")

        # After the robot has started...
        try:
            place_task = PlaceOnReceptacleTask(
                self.agent,
                target_receptacle=target_receptacle,
                matching=self._match_method,
            )
            task = place_task.get_task(add_rotate=True)
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def _find(self, target_object: str) -> None:
        """Create a task to find the object and execute it.

        Args:
            target_object: The object to find.
        """

        logger.alert(f"[Find task] Find: {target_object}")

        # After the robot has started...
        try:
            find_task = FindObjectTask(
                self.agent,
                target_object=target_object,
                matching=self._match_method,
            )
            task = find_task.get_task(add_rotate=True)
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def _hand_over(self) -> None:
        """Create a task to find a person, navigate to them, and extend the arm toward them"""
        logger.alert(f"[Pickup task] Hand Over")

        # After the robot has started...
        try:
            hand_over_task = HandOverTask(self.agent)
            task = hand_over_task.get_task()
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def __call__(self, response: List[Tuple[str, str]], channel=None) -> bool:
        """Execute the list of commands given by the LLM bot.

        Args:
            response: A list of tuples, where the first element is the command and the second is the argument.
            channel (Optional): The discord channel to send messages to, if using discord bot.

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
                if channel is not None:
                    # obs = self.robot.get_observation()
                    # self.discord_bot.send_message(channel=channel, message=args, content=numpy_image_to_bytes(obs.rgb))
                    # Optionally strip quotes from args
                    if args[0] == '"' and args[-1] == '"':
                        args = args[1:-1]
                    self.discord_bot.send_message(channel=channel, message=args)
                self.agent.robot_say(args)
            elif command == "pickup":
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args
                i += 1
                if i >= len(response):
                    logger.warning(
                        "Pickup without place! Try giving a full pick-and-place instruction."
                    )
                    self._pickup(target_object, None)
                    # Continue works here because we've already incremented i
                    continue
                next_command, next_args = response[i]
                if next_command != "place":
                    logger.warning(
                        "Pickup without place! Try giving a full pick-and-place instruction."
                    )
                    self._pickup(target_object, None)
                    # Continue works here because we've already incremented i
                    continue
                else:
                    logger.info(f"{i} {next_command} {next_args}")
                    logger.info(f"[Pickup task] Place: {next_args}")
                target_receptacle = next_args
                self._pickup(target_object, target_receptacle)
            elif command == "place":
                logger.warning(
                    "Place without pickup! Try giving a full pick-and-place instruction."
                )
                self._place(args)
            elif command == "hand_over":
                self._hand_over()
            elif command == "take_picture":
                self._take_picture(channel)
            elif command == "take_ee_picture":
                self._take_ee_picture(channel)
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
