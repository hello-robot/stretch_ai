# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Optional

from termcolor import colored

from stretch.agent.robot_agent import RobotAgent
from stretch.core.robot import AbstractRobotClient
from stretch.core.task import Operation
from stretch.mapping.instance import Instance


class ManagedOperation(Operation):
    """Placeholder node for an example in a task plan. Contains some functions to make it easier to print out different messages with color for interpretability, and also provides some utilities for making the robot do different tasks."""

    def __init__(
        self,
        name,
        agent: Optional[RobotAgent] = None,
        robot: Optional[AbstractRobotClient] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        if agent is not None:
            self.agent = agent
            self.robot = agent.robot
            self.parameters = agent.parameters
            self.navigation_space = agent.space
        elif robot is not None:
            self.robot = robot
            self.parameters = robot.parameters
        else:
            raise ValueError("Either agent or robot must be provided.")

        assert self.robot is not None, "Robot client must be provided."

        # Get the robot kinematic model
        self.robot_model = self.robot.get_robot_model()

        # Determine whether machine is headless
        self.headless_machine = "DISPLAY" not in os.environ

    @property
    def name(self) -> str:
        """Return the name of the operation.

        Returns:
            str: the name of the operation
        """
        return self._name

    def update(self, **kwargs):
        print(colored("================ Updating the world model ==================", "blue"))
        self.agent.update(**kwargs)

    def attempt(self, message: str):
        print(colored(f"Trying {self.name}:", "blue"), message)

    def intro(self, message: str):
        print(colored(f"Running {self.name}:", "green"), message)

    def warn(self, message: str):
        print(colored(f"Warning in {self.name}: {message}", "yellow"))

    def error(self, message: str):
        print(colored(f"Error in {self.name}: {message}", "red"))

    def info(self, message: str):
        print(colored(f"{self.name}: {message}", "blue"))

    def cheer(self, message: str):
        """An upbeat message!"""
        print(colored(f"!!! {self.name} !!!: {message}", "green"))

    def show_instance_segmentation_image(self):
        # Show the last instance image
        import matplotlib

        # TODO: why do we need to configure this every time
        if self.headless_machine:
            matplotlib.use("Agg")
        else:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.imshow(self.agent.get_voxel_map().observations[0].instance)
        plt.show()

    def show_instance(self, instance: Instance, title: Optional[str] = None):
        """Show the instance in the voxel grid."""
        import matplotlib

        if self.headless_machine:
            matplotlib.use("Agg")
        else:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        view = instance.get_best_view()
        plt.imshow(view.get_image())
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()
