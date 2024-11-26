# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Optional, Tuple

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

        # Other parameters
        self.visual_servo = visual_servo
        self.match_method = match_method

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")

        # Configuration
        self._match_method = match_method

        # Create semantic sensor if visual servoing is enabled
        print("- Create semantic sensor if visual servoing is enabled")
        if self.visual_servo:
            self.semantic_sensor = create_semantic_sensor(
                parameters=self.parameters,
                device_id=device_id,
                verbose=False,
            )
        else:
            self.parameters["encoder"] = None
            self.semantic_sensor = None

        print("- Start robot agent with data collection")
        self.agent = RobotAgent(self.robot, self.parameters, self.semantic_sensor)
        self.agent.start()

        # Create grasp object operation
        if self.visual_servo:
            self.grasp_object = GraspObjectOperation(
                "grasp_the_object",
                self.agent,
            )
        else:
            self.grasp_object = None

        # Task stuff
        self.emote_task = EmoteTask(self.agent)

    def _find(self, target_object: str) -> np.ndarray:
        """Find an object. This is a helper function for the main loop.

        Args:
            target_object: The object to find.

        Returns:
            The point where the object is located.
        """
        self.robot.move_to_nav_posture()
        self.robot.switch_to_navigation_mode()
        point = self.agent.navigate(target_object)
        if point is None:
            logger.error("Navigation Failure: Could not find the object {}".format(target_object))
        cv2.imwrite(target_object + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)
        return point

    def _pickup(
        self,
        target_object: str,
        point: Optional[np.ndarray] = None,
        skip_confirmations: bool = False,
    ) -> None:
        """Pick up an object.

        Args:
            target_object: The object to pick up.
        """
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6

        # Grasp the object using operation if it's available
        if self.grasp_object is not None:
            self.robot.say("Grasping the " + str(target_object) + ".")
            print("Using operation to grasp object:", target_object)
            print(" - Point:", point)
            print(" - Theta:", theta)
            self.grasp_object(
                target_object=target_object,
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
            print("Using self.agent to grasp object:", target_object)
            self.agent.manipulate(target_object, theta, skip_confirmation=skip_confirmations)
        self.robot.look_front()

    def _place(self, target_receptacle: str, point: Optional[np.ndarray]) -> None:
        """Place an object.

        Args:
            target_receptacle: The receptacle to place the object in.
        """
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6

        self.robot.say("Placing object on the " + str(target_receptacle) + ".")
        self.agent.place(target_receptacle, theta)
        self.robot.move_to_nav_posture()

    def _navigate_to(self, target_receptacle: str) -> np.ndarray:
        """Navigate to a receptacle.

        Args:
            target_receptacle: The receptacle to navigate to.
        """

        self.robot.switch_to_navigation_mode()
        print("Going to the " + str(target_receptacle) + ".")
        point = self.agent.navigate(target_receptacle)

        if point is None:
            print("Navigation Failure")
            self.robot.say("I could not find the " + str(target_receptacle) + ".")

        cv2.imwrite(target_receptacle + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)

        return point

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
                point = self._find(args)
                self._pickup(target_object, point=point)
            elif command == "place":
                logger.info(f"[Pickup task] Place: {args}")
                point = self._navigate_to(args)
                self._place(args, point=point)
            elif command == "wave":
                logger.info("[Pickup task] Waving.")
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()
            elif command == "go_home":
                logger.info("[Pickup task] Going home.")
                if self.agent.get_voxel_map().is_empty():
                    logger.warning("No map data available. Cannot go home.")
                else:
                    self.agent.go_home()
            elif command == "explore":
                logger.info("[Pickup task] Exploring.")
                self.agent.explore()
            elif command == "find":
                logger.info("[Pickup task] Finding {}.".format(args))
                point = self._find(args)
            elif command == "nod_head":
                logger.info("[Pickup task] Nodding head.")
                self.emote_task.get_task("nod_head").run()
            elif command == "shake_head":
                logger.info("[Pickup task] Shaking head.")
                self.emote_task.get_task("shake_head").run()
            elif command == "avert_gaze":
                logger.info("[Pickup task] Averting gaze.")
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
        mode: str = None,
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

        if mode is None:
            mode = get_mode(mode)

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
                    if target_object is not None:
                        text = target_object
                    else:
                        text = input("Enter object name: ")
                    point = self._find(text)

                # If the object is found, grasp it
                if skip_confirmations or input("Do you want to pick up an object? (y/n): ") != "n":
                    if text is None:
                        text = input("Enter object name: ")
                    self._pickup(text, point=point, skip_confirmations=skip_confirmations)

                # Reset text and point for placement
                text = None
                point = None
                if skip_confirmations or input("You want to find a receptacle? (y/n): ") != "n":
                    if target_receptacle is not None:
                        text = target_receptacle
                    else:
                        text = input("Enter receptacle name: ")
                    point = self._navigate_to(text)

                # Execute placement if the object is found
                if skip_confirmations or input("You want to run placement? (y/n): ") != "n":
                    if text is None:
                        text = input("Enter receptacle name: ")
                    self._place(text, point=point)

                self.agent.voxel_map.write_to_pickle()

            # Clear mode after the first trial - otherwise it will go on forever
            mode = None
