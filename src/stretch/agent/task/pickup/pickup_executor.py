# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Tuple

from stretch.agent.operations import (
    AvertGazeOperation,
    ExploreOperation,
    GoHomeOperation,
    GoToNavOperation,
    GraspObjectOperation,
    NavigateToObjectOperation,
    NodHeadOperation,
    OpenLoopGraspObjectOperation,
    PlaceObjectOperation,
    PreGraspObjectOperation,
    RotateInPlaceOperation,
    SearchForObjectOnFloorOperation,
    SearchForReceptacleOperation,
    ShakeHeadOperation,
    SpeakOperation,
    WaveOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.agent.task.pickup.pickup_task import PickupTask
from stretch.core import AbstractRobotClient
from stretch.core.task import Task
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
        available_actions: List[str],
        match_method: str = "feature",
        open_loop: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the executor.

        Args:
            robot: The robot client.
            agent: The robot agent.
            dry_run: If true, don't actually execute the commands.
        """
        self.robot = robot
        self.agent = agent

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

        # Managed task
        self._task = Task()
        self._operation_count = 0
        self.available_actions = available_actions

    def say(self, text: str) -> None:
        """Use the robot to say the text."""
        say_operation = SpeakOperation(
            f"{str(self._operation_count)}_say_" + text, agent=self.agent, robot=self.robot
        )
        say_operation.configure(message=text)
        self._task.add_operation(say_operation, True)
        self._operation_count += 1

    def pickup(self, target_object: str, target_receptacle: str) -> None:
        """Pick up the object and place it in the receptacle."""
        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation(
            f"{str(self._operation_count)}_go to navigation mode", self.agent, retry_on_failure=True
        )
        self._operation_count += 1

        # Spin in place to find objects.
        rotate_in_place = RotateInPlaceOperation(
            f"{str(self._operation_count)}_rotate_in_place",
            self.agent,
            parent=go_to_navigation_mode,
        )
        self._operation_count += 1

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacleOperation(
            f"{str(self._operation_count)}_search_for_{self.target_receptacle}",
            self.agent,
            parent=rotate_in_place,
            retry_on_failure=True,
            match_method=self._match_method,
        )
        self._operation_count += 1

        # Try to expand the frontier and find an object; or just wander around for a while.
        search_for_object = SearchForObjectOnFloorOperation(
            f"{str(self._operation_count)}_search_for_{self.target_object}_on_floor",
            self.agent,
            retry_on_failure=True,
            match_method=self._match_method,
        )
        self._operation_count += 1
        if self.agent.target_object is not None:
            # Overwrite the default object to search for
            search_for_object.set_target_object_class(self.agent.target_object)
        if self.agent.target_receptacle is not None:
            search_for_receptacle.set_target_object_class(self.agent.target_receptacle)

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            f"{str(self._operation_count)}_go_to_object",
            self.agent,
            parent=search_for_object,
            on_cannot_start=search_for_object,
            to_receptacle=False,
        )
        self._operation_count += 1

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_receptacle = NavigateToObjectOperation(
            f"{str(self._operation_count)}_go_to_receptacle",
            self.agent,
            on_cannot_start=search_for_receptacle,
            to_receptacle=True,
        )
        self._operation_count += 1

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        # If we cannot find the object, we should go back to the search_for_object operation.
        # To determine if we can start, we just check to see if there's a detectable object nearby.
        pregrasp_object = PreGraspObjectOperation(
            f"{str(self._operation_count)}_prepare_to_grasp",
            self.agent,
            on_failure=None,
            on_cannot_start=go_to_object,
            retry_on_failure=True,
        )
        self._operation_count += 1
        # If we cannot start, we should go back to the search_for_object operation.
        # To determine if we can start, we look at the end effector camera and see if there's anything detectable.
        if self.use_visual_servoing_for_grasp:
            grasp_object = GraspObjectOperation(
                f"{str(self._operation_count)}_grasp_the_{self.target_object}",
                self.agent,
                parent=pregrasp_object,
                on_failure=pregrasp_object,
                on_cannot_start=go_to_object,
                retry_on_failure=False,
            )
            grasp_object.set_target_object_class(self.agent.target_object)
            grasp_object.servo_to_grasp = True
            grasp_object.match_method = self._match_method
            self._operation_count += 1
        else:
            grasp_object = OpenLoopGraspObjectOperation(
                f"{str(self._operation_count)}_grasp_the_{self.target_object}",
                self.agent,
                parent=pregrasp_object,
                on_failure=pregrasp_object,
                on_cannot_start=go_to_object,
                retry_on_failure=False,
            )
            grasp_object.set_target_object_class(self.agent.target_object)
            grasp_object.match_method = self._match_method
            self._operation_count += 1

        place_object_on_receptacle = PlaceObjectOperation(
            f"{str(self._operation_count)}_place_object_on_receptacle",
            self.agent,
            on_cannot_start=go_to_receptacle,
        )
        self._operation_count += 1

        self._task.add_operation(go_to_navigation_mode)
        self._task.add_operation(rotate_in_place)
        self._task.add_operation(search_for_receptacle)
        self._task.add_operation(search_for_object)
        self._task.add_operation(go_to_object)
        self._task.add_operation(pregrasp_object)
        self._task.add_operation(grasp_object)
        self._task.add_operation(go_to_receptacle)
        self._task.add_operation(place_object_on_receptacle)

        self._task.connect_on_success(go_to_navigation_mode.name, search_for_receptacle.name)
        self._task.connect_on_success(search_for_receptacle.name, search_for_object.name)
        self._task.connect_on_success(search_for_object.name, go_to_object.name)
        self._task.connect_on_success(go_to_object.name, pregrasp_object.name)
        self._task.connect_on_success(pregrasp_object.name, grasp_object.name)
        self._task.connect_on_success(grasp_object.name, go_to_receptacle.name)
        self._task.connect_on_success(go_to_receptacle.name, place_object_on_receptacle.name)

        self._task.connect_on_success(search_for_receptacle.name, search_for_object.name)

        self._task.connect_on_cannot_start(go_to_object.name, search_for_object.name)

    def wave(self) -> None:
        """Wave to the user."""
        wave_operation = WaveOperation(
            f"{str(self._operation_count)}_wave", self.agent, robot=self.robot
        )
        self._task.add_operation(wave_operation, True)
        self._operation_count += 1

    def go_home(self) -> None:
        """Go back to the home position."""
        go_home_operation = GoHomeOperation(
            f"{str(self._operation_count)}_go_home", self.agent, robot=self.robot
        )
        self._task.add_operation(go_home_operation, True)
        self._operation_count += 1

    def explore(self) -> None:
        """Explore the environment."""
        explore_operation = ExploreOperation(
            f"{str(self._operation_count)}_explore", self.agent, robot=self.robot
        )
        self._task.add_operation(explore_operation, True)
        self._operation_count += 1

    def nod_head(self) -> None:
        """Nod the head."""
        nod_head_operation = NodHeadOperation(
            f"{str(self._operation_count)}_nod_head", self.agent, robot=self.robot
        )
        self._task.add_operation(nod_head_operation, True)
        self._operation_count += 1

    def shake_head(self) -> None:
        """Shake the head."""
        shake_head_operation = ShakeHeadOperation(
            f"{str(self._operation_count)}_shake_head", self.agent, robot=self.robot
        )
        self._task.add_operation(shake_head_operation, True)
        self._operation_count += 1

    def avert_gaze(self) -> None:
        """Avert the gaze."""
        avert_gaze_operation = AvertGazeOperation(
            f"{str(self._operation_count)}_avert_gaze", self.agent, robot=self.robot
        )
        self._task.add_operation(avert_gaze_operation, True)
        self._operation_count += 1

    def find(self, object_name: str) -> None:
        """Find the object."""
        speak_not_implemented = SpeakOperation(
            f"{str(self._operation_count)}_find_" + object_name, agent=self.agent, robot=self.robot
        )
        speak_not_implemented.configure(message="Find operation not implemented")
        self._task.add_operation(speak_not_implemented, True)
        self._operation_count += 1

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
        self._task = Task()
        self._operation_count = 0

        if response is None or len(response) == 0:
            logger.error("No commands to execute!")
            self.agent.robot_say("I'm sorry, I didn't understand that.")
            return True

        # Loop over every command we have been given
        # Pull out pickup and place as a single arg if they are in a row
        # Else, execute things as they come
        while i < len(response):
            command, args = response[i]
            logger.info(f"{i} {command} {args}")

            if command in self.available_actions:
                command_with_args = "self." + command + "(" + args + ")"

                if command == "pickup":
                    if (i + 1 < len(response)) and (response[i + 1][0] == "place"):
                        eval(command_with_args)
                        i += 1
                    else:
                        logger.error("Pickup without place! Doing nothing.")
                else:
                    eval(command_with_args)
            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1

        self._task.run()

        return True
