# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np

from stretch.agent.operations import (
    ExtendArm,
    NavigateToObjectOperation,
    OpenGripper,
    SpeakOperation,
    UpdateOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task


class HandOverTask:
    """Find a person, navigate to them, and extend the arm toward them"""

    def __init__(self, agent: RobotAgent, target_object: str = "person") -> None:
        # super().__init__(agent)
        self.agent = agent

        self.target_object = target_object

        # Sync these things
        self.robot = self.agent.robot
        self.voxel_map = self.agent.get_voxel_map()
        self.navigation_space = self.agent.space
        self.semantic_sensor = self.agent.semantic_sensor
        self.parameters = self.agent.parameters
        self.instance_memory = self.agent.get_voxel_map().instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        self.current_receptacle = None
        self.agent.reset_object_plans()

    def get_task(self, add_rotate: bool = False, mode: str = "one_shot") -> Task:
        """Create a task plan with loopbacks and recovery from failure. The robot will explore the environment, find objects, and pick them up

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.
            mode (str, optional): Type of task to create. Can be "one_shot" or "all". Defaults to "one_shot".

        Returns:
            Task: Executable task plan for the robot to pick up objects in the environment.
        """

        return self.get_one_shot_task(add_rotate=add_rotate)

    def get_one_shot_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan"""

        task = Task()

        update = UpdateOperation("update_scene", self.agent, retry_on_failure=True)
        update.configure(
            move_head=True,
            target_object=self.target_object,
            show_map_so_far=False,  # Uses Open3D display (blocking)
            clear_voxel_map=False,  # True,
            show_instances_detected=False,  # Uses OpenCV image display (blocking)
            match_method="name",  # "feature",
            arm_height=0.6,
            tilt=-1.0 * np.pi / 8.0,
        )

        found_a_person = SpeakOperation(
            name="found_a_person", agent=self.agent, parent=update, on_cannot_start=update
        )
        found_a_person.configure(
            message="I found a person! I am going to navigate to them.", sleep_time=2.0
        )

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            name="go_to_object",
            agent=self.agent,
            parent=found_a_person,
            on_cannot_start=update,
            to_receptacle=False,
        )

        ready_to_extend_arm = SpeakOperation(
            name="ready_to_extend_arm",
            agent=self.agent,
            parent=go_to_object,
            on_cannot_start=update,
        )
        ready_to_extend_arm.configure(
            message="I navigated to the person. Now I am going to extend my arm toward them.",
            sleep_time=2.0,
        )

        extend_arm = ExtendArm(
            name="extend_arm",
            agent=self.agent,
            parent=ready_to_extend_arm,
            on_cannot_start=update,
        )
        extend_arm.configure(arm_extension=0.2)

        ready_to_open_gripper = SpeakOperation(
            name="ready_to_open_gripper",
            agent=self.agent,
            parent=extend_arm,
            on_cannot_start=update,
        )
        ready_to_open_gripper.configure(
            message="I am now going to open my gripper to release the object.", sleep_time=2.0
        )

        open_gripper = OpenGripper(
            name="open_gripper",
            agent=self.agent,
            parent=ready_to_open_gripper,
            on_cannot_start=update,
        )

        finished_handover = SpeakOperation(
            name="finished_handover", agent=self.agent, parent=open_gripper, on_cannot_start=update
        )
        finished_handover.configure(
            message="I have finished handing over the object.", sleep_time=5.0
        )

        retract_arm = ExtendArm(
            name="retract_arm",
            agent=self.agent,
            parent=finished_handover,
            on_cannot_start=update,
        )
        retract_arm.configure(arm_extension=0.05)

        # task.add_operation(go_to_navigation_mode)
        # if add_rotate:
        #    task.add_operation(rotate_in_place)
        task.add_operation(update)
        task.add_operation(found_a_person)
        task.add_operation(go_to_object)
        task.add_operation(ready_to_extend_arm)
        task.add_operation(extend_arm)
        task.add_operation(ready_to_open_gripper)
        task.add_operation(open_gripper)
        task.add_operation(finished_handover)
        task.add_operation(retract_arm)

        # Terminate on a successful place
        return task


if __name__ == "__main__":
    from stretch.agent.robot_agent import RobotAgent
    from stretch.agent.zmq_client import HomeRobotZmqClient

    robot = HomeRobotZmqClient()
    # Create a robot agent with instance memory
    agent = RobotAgent(robot, create_semantic_sensor=True)

    task = HandOverTask(agent).get_task(add_rotate=False)
    task.run()
