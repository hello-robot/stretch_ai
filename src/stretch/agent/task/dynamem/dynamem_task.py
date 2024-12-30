# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from stretch.agent.operations import GraspObjectOperation
from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.agent.task.pickup.hand_over_task import HandOverTask
from stretch.core import AbstractRobotClient, Parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.image import numpy_image_to_bytes

# Mapping and perception
from stretch.utils.logger import Logger

logger = Logger(__name__)


def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


class DynamemTaskExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters,
        match_method: str = "feature",
        visual_servo: bool = False,
        device_id: int = 0,
        output_path: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        skip_confirmations: bool = True,
        explore_iter: int = 5,
        mllm: bool = False,
        manipulation_only: bool = False,
        discord_bot=None,
    ) -> None:
        """Initialize the executor."""
        self.robot = robot
        self.parameters = parameters
        self.discord_bot = discord_bot

        # Other parameters
        self.visual_servo = visual_servo
        self.match_method = match_method
        self.skip_confirmations = skip_confirmations
        self.explore_iter = explore_iter

        self.manipulation_only = manipulation_only

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")

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
        self.agent = RobotAgent(
            self.robot,
            self.parameters,
            self.semantic_sensor,
            log=output_path,
            server_ip=server_ip,
            mllm=mllm,
            manipulation_only=manipulation_only,
        )
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
        self.robot.switch_to_navigation_mode()
        point = self.agent.navigate(target_object)
        # `filename` = None means write to default log path (the datetime you started to run the process)
        self.agent.voxel_map.write_to_pickle(filename=None)
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
            state = self.robot.get_six_joints()
            state[1] = 1.0
            self.robot.arm_to(state, blocking=True)
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
        # If you run this stack with visual servo, run it locally
        self.agent.place(target_receptacle, init_tilt=theta, local=self.visual_servo)
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

    def __call__(self, response: List[Tuple[str, str]], channel=None) -> bool:
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
                if channel is not None:
                    # Optionally strip quotes from args
                    if args[0] == '"' and args[-1] == '"':
                        args = args[1:-1]
                    self.discord_bot.send_message(channel=channel, message=args)
            elif command == "pickup":
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args
                next_command, next_args = response[i]

                # Navigation

                # Either we wait for users to confirm whether to run navigation, or we just directly control the robot to navigate.
                if self.skip_confirmations or (
                    not self.skip_confirmations
                    and input("Do you want to run navigation? [Y/n]: ").upper() != "N"
                ):
                    self.robot.move_to_nav_posture()
                    point = self._find(args)
                # Or the user explicitly tells that he or she does not want to run navigation.
                else:
                    point = None

                # Pick up

                if self.skip_confirmations:
                    if point is not None:
                        self._pickup(target_object, point=point)
                    else:
                        logger.error("Could not find the object.")
                        self.robot.say("I could not find the " + str(args) + ".")
                        i += 1
                        continue
                else:
                    if input("Do you want to run picking? [Y/n]: ").upper() != "N":
                        self._pickup(target_object, point=point)
                    else:
                        logger.info("Skip picking!")
                        i += 1
                        continue

            elif command == "place":
                logger.info(f"[Pickup task] Place: {args}")
                target_object = args
                next_command, next_args = response[i]

                # Navigation

                # Either we wait for users to confirm whether to run navigation, or we just directly control the robot to navigate.
                if self.skip_confirmations or (
                    not self.skip_confirmations
                    and input("Do you want to run navigation? [Y/n]: ").upper() != "N"
                ):
                    point = self._find(args)
                # Or the user explicitly tells that he or she does not want to run navigation.
                else:
                    point = None

                # Placing

                if self.skip_confirmations:
                    if point is not None:
                        self._place(target_object, point=point)
                    else:
                        logger.error("Could not find the object.")
                        self.robot.say("I could not find the " + str(args) + ".")
                        i += 1
                        continue
                else:
                    if input("Do you want to run placement? [Y/n]: ").upper() != "N":
                        self._place(target_object, point=point)
                    else:
                        logger.info("Skip placing!")
                        i += 1
                        continue
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
                # `filename` = None means write to default log path (the datetime you started to run the process)
                self.agent.voxel_map.write_to_pickle(filename=None)
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
                for _ in range(self.explore_iter):
                    self.agent.run_exploration()
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
            elif command == "take_picture":
                self._take_picture(channel)
            elif command == "take_ee_picture":
                self._take_ee_picture(channel)
            elif command == "end":
                logger.info("[Pickup task] Ending.")
                break
            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1
        # If we did not explicitly receive a quit command, we are not yet done.
        return True
