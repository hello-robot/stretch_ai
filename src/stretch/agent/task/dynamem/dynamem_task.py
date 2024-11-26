# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple

from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.core import AbstractRobotClient

# Mapping and perception
from stretch.utils.logger import Logger

logger = Logger(__name__)


class DynamemTaskExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters,
        match_method: str = "feature",
    ) -> None:
        """Initialize the executor.

        Args:
            robot: The robot client.
            agent: The robot agent.
            dry_run: If true, don't actually execute the commands.
        """
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
    
    def _navigate_to(self, target_receptacle: str) -> None:
        """Navigate to a receptacle.

        Args:
            target_receptacle: The receptacle to navigate to.
        """

        self.robot.switch_to_navigation_mode()
        print("Going to the " + str(target_receptacle) + ".")
        point = agent.navigate(target_receptacle)

        if point is None:
            print("Navigation Failure")
            self.robot.say("I could not find the " + str(target_receptacle) + ".")

        cv2.imwrite(target_receptacle + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)



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


def main():

    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    if visual_servo:
        grasp_object = GraspObjectOperation(
            "grasp_the_object",
            agent,
        )
    else:
        grasp_object = None

    if input_path is None:
        agent.rotate_in_place()
    else:
        agent.voxel_map.read_from_pickle(input_path)

    agent.voxel_map.write_to_pickle()

    while agent.is_running():

        # If target object and receptacle are provided, set mode to manipulation
        if target_object is not None and target_receptacle is not None:
            mode = "M"
        else:
            # Get mode from user input
            mode = get_mode(mode)

        if mode == "S":
            robot.say("Saving data. Goodbye!")
            agent.voxel_map.write_to_pickle()
            break

        if mode == "E":
            robot.switch_to_navigation_mode()
            robot.say("Exploring.")
            for epoch in range(explore_iter):
                print("\n", "Exploration epoch ", epoch, "\n")
                if not agent.run_exploration():
                    print("Exploration failed! Quitting!")
                    continue
        else:
            # Add some audio to make it easier to tell what's going on.
            robot.say("Running manipulation.")

            text = None
            point = None

            if skip_confirmations or input("Do you want to look for an object? (y/n): ") != "n":
                robot.move_to_nav_posture()
                robot.switch_to_navigation_mode()
                if target_object is not None:
                    text = target_object
                else:
                    text = input("Enter object name: ")
                point = agent.navigate(text)
                if point is None:
                    print("Navigation Failure!")
                cv2.imwrite(text + ".jpg", robot.get_observation().rgb[:, :, [2, 1, 0]])
                robot.switch_to_navigation_mode()
                xyt = robot.get_base_pose()
                xyt[2] = xyt[2] + np.pi / 2
                robot.move_base_to(xyt, blocking=True)

            # If the object is found, grasp it
            if skip_confirmations or input("Do you want to pick up an object? (y/n): ") != "n":
                robot.switch_to_manipulation_mode()
                if text is None:
                    text = input("Enter object name: ")
                camera_xyz = robot.get_head_pose()[:3, 3]
                if point is not None:
                    theta = compute_tilt(camera_xyz, point)
                else:
                    theta = -0.6

                # Grasp the object using operation if it's available
                if grasp_object is not None:
                    robot.say("Grasping the " + str(text) + ".")
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
                    robot.move_to_nav_posture()
                else:
                    # Otherwise, use the agent's manipulation method
                    # This is from OK Robot
                    print("Using agent to grasp object:", text)
                    agent.manipulate(text, theta, skip_confirmation=skip_confirmations)
                robot.look_front()

            # Reset text and point for placement
            text = None
            point = None
            if skip_confirmations or input("You want to find a receptacle? (y/n): ") != "n":
                if target_receptacle is not None:
                    text = target_receptacle
                else:
                    text = input("Enter receptacle name: ")

                self._navigate_to(text)

            # Execute placement if the object is found
            if skip_confirmations or input("You want to run placement? (y/n): ") != "n":
                robot.switch_to_manipulation_mode()

                if text is None:
                    text = input("Enter receptacle name: ")

                camera_xyz = robot.get_head_pose()[:3, 3]
                if point is not None:
                    theta = compute_tilt(camera_xyz, point)
                else:
                    theta = -0.6

                robot.say("Placing object on the " + str(text) + ".")
                agent.place(text, theta)
                robot.move_to_nav_posture()

            agent.voxel_map.write_to_pickle()

        # Clear mode after the first trial - otherwise it will go on forever
        mode = None
