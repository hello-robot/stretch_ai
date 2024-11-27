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
from stretch.agent.task.pickup.hand_over_task import HandOverTask
from stretch.core import AbstractRobotClient, Parameters
from stretch.dynav.utils import compute_tilt
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
        output_path: Optional[str] = None,
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
        self.agent = RobotAgent(self.robot, self.parameters, self.semantic_sensor, log=output_path)
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
            return None
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
            return None

        print("Saving current robot memory to pickle file")
        self.agent.voxel_map.write_to_pickle()
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

        # Dynamem aims to life long robot, we should not reset the robot's memory.
        # logger.info("Resetting agent...")
        # self.agent.reset()

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
                if point is not None:
                    self._pickup(target_object, point=point)
                else:
                    logger.error("Could not find the object.")
                    self.robot.say("I could not find the " + str(args) + ".")
                    break
            elif command == "place":
                logger.info(f"[Pickup task] Place: {args}")
                point = self._navigate_to(args)
                if point is not None:
                    self._place(args, point=point)
                else:
                    logger.error("Could not navigate to the receptacle.")
                    self.robot.say("I could not find the " + str(args) + ".")
                    break
            elif command == "hand_over":
                self._hand_over()
            elif command == "wave":
                logger.info("[Pickup task] Waving.")
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()
            elif command == "rotate_in_place":
                logger.info("Rotate in place to scan environments.")
                self.agent.rotate_in_place()
                self.agent.voxel_map.write_to_pickle()
            elif command == "read_from_pickle":
                logger.info(f"Load the semantic memory from past runs, pickle file name: {args}.")
                self.agent.voxel_map.read_from_pickle(args)
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
