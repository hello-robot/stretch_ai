# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple

import cv2
import numpy as np

from stretch.agent.operations import GraspObjectOperation
from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.core import AbstractRobotClient, Parameters
from stretch.dynav.utils import compute_tilt, get_mode
from stretch.perception import create_semantic_sensor

# Mapping and perception
from stretch.utils.logger import Logger

logger = Logger(__name__)


class DynamemTaskExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters,
        match_method: str = "feature",
        visual_servo: bool = False,
        device_id: int = 0,
    ) -> None:
        """Initialize the executor."""
        self.robot = robot
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

    def run(
        self,
        mode: str,
        target_object: str = None,
        target_receptacle: str = None,
        input_path: str = None,
        skip_confirmations: bool = False,
        explore_iter: int = 3,
    ) -> None:
        """This is the core logic from the original run_dynamem.py script."""

        object_to_find, location_to_place = None, None
        self.robot.move_to_nav_posture()
        self.robot.set_velocity(v=30.0, w=15.0)

        if self.visual_servo:
            grasp_object = GraspObjectOperation(
                "grasp_the_object",
                self.agent,
            )
        else:
            grasp_object = None

        if input_path is None:
            self.agent.rotate_in_place()
        else:
            self.agent.voxel_map.read_from_pickle(input_path)

        self.agent.voxel_map.write_to_pickle()

        while self.agent.is_running():

            # If target object and receptacle are provided, set mode to manipulation
            if target_object is not None and target_receptacle is not None:
                mode = "M"
            else:
                # Get mode from user input
                mode = get_mode(mode)

            if mode == "S":
                self.robot.say("Saving data. Goodbye!")
                self.agent.voxel_map.write_to_pickle()
                break

            if mode == "E":
                self.robot.switch_to_navigation_mode()
                self.robot.say("Exploring.")
                for epoch in range(explore_iter):
                    print("\n", "Exploration epoch ", epoch, "\n")
                    if not self.agent.run_exploration():
                        print("Exploration failed! Quitting!")
                        continue
            else:
                # Add some audio to make it easier to tell what's going on.
                self.robot.say("Running manipulation.")

                text = None
                point = None

                if skip_confirmations or input("Do you want to look for an object? (y/n): ") != "n":
                    self.robot.move_to_nav_posture()
                    self.robot.switch_to_navigation_mode()
                    if target_object is not None:
                        text = target_object
                    else:
                        text = input("Enter object name: ")
                    point = self.agent.navigate(text)
                    if point is None:
                        print("Navigation Failure!")
                    cv2.imwrite(text + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
                    self.robot.switch_to_navigation_mode()
                    xyt = self.robot.get_base_pose()
                    xyt[2] = xyt[2] + np.pi / 2
                    self.robot.move_base_to(xyt, blocking=True)

                # If the object is found, grasp it
                if skip_confirmations or input("Do you want to pick up an object? (y/n): ") != "n":
                    self.robot.switch_to_manipulation_mode()
                    if text is None:
                        text = input("Enter object name: ")
                    camera_xyz = self.robot.get_head_pose()[:3, 3]
                    if point is not None:
                        theta = compute_tilt(camera_xyz, point)
                    else:
                        theta = -0.6

                    # Grasp the object using operation if it's available
                    if grasp_object is not None:
                        self.robot.say("Grasping the " + str(text) + ".")
                        print("Using operation to grasp object:", text)
                        print(" - Point:", point)
                        print(" - Theta:", theta)
                        grasp_object(
                            target_object=text,
                            object_xyz=point,
                            match_method="feature",
                            show_object_to_grasp=False,
                            show_servo_gui=True,
                            delete_object_after_grasp=False,
                        )
                        # This retracts the arm
                        self.robot.move_to_nav_posture()
                    else:
                        # Otherwise, use the self.agent's manipulation method
                        # This is from OK Robot
                        print("Using self.agent to grasp object:", text)
                        self.agent.manipulate(text, theta, skip_confirmation=skip_confirmations)
                    self.robot.look_front()

                # Reset text and point for placement
                text = None
                point = None
                if skip_confirmations or input("You want to find a receptacle? (y/n): ") != "n":
                    self.robot.switch_to_navigation_mode()
                    if target_receptacle is not None:
                        text = target_receptacle
                    else:
                        text = input("Enter receptacle name: ")

                    print("Going to the " + str(text) + ".")
                    point = self.agent.navigate(text)

                    if point is None:
                        print("Navigation Failure")
                        self.robot.say("I could not find the " + str(text) + ".")

                    cv2.imwrite(text + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
                    self.robot.switch_to_navigation_mode()
                    xyt = self.robot.get_base_pose()
                    xyt[2] = xyt[2] + np.pi / 2
                    self.robot.move_base_to(xyt, blocking=True)

                # Execute placement if the object is found
                if skip_confirmations or input("You want to run placement? (y/n): ") != "n":
                    self.robot.switch_to_manipulation_mode()

                    if text is None:
                        text = input("Enter receptacle name: ")

                    camera_xyz = self.robot.get_head_pose()[:3, 3]
                    if point is not None:
                        theta = compute_tilt(camera_xyz, point)
                    else:
                        theta = -0.6

                    self.robot.say("Placing object on the " + str(text) + ".")
                    self.agent.place(text, theta)
                    self.robot.move_to_nav_posture()

                self.agent.voxel_map.write_to_pickle()

            # Clear mode after the first trial - otherwise it will go on forever
            mode = None
