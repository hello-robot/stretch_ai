from typing import Optional

import numpy as np
from termcolor import colored

from stretch.core.task import Operation
from stretch.mapping.instance import Instance


class ManagedOperation(Operation):
    """Placeholder node for an example in a task plan. Contains some functions to make it easier to print out different messages with color for interpretability, and also provides some utilities for making the robot do different tasks."""

    def __init__(self, name, manager, **kwargs):
        super().__init__(name, **kwargs)
        self.manager = manager
        self.robot = manager.robot
        self.parameters = manager.parameters
        self.navigation_space = manager.navigation_space
        self.agent = manager.agent
        self.robot_model = self.robot.get_robot_model()

    @property
    def name(self) -> str:
        """Return the name of the operation.

        Returns:
            str: the name of the operation
        """
        return self._name

    def update(self):
        print(colored("================ Updating the world model ==================", "blue"))
        self.agent.update()

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

    def plan_to_instance_for_manipulation(self, instance, start, radius_m: float = 0.45):
        """Manipulation planning wrapper. Plan to instance with a radius around it, ensuring a base location can be found in explored space."""
        return self.agent.plan_to_instance(
            instance, start=start, rotation_offset=np.pi / 2, radius_m=radius_m, max_tries=100
        )

    def show_instance_segmentation_image(self):
        # Show the last instance image
        import matplotlib

        # TODO: why do we need to configure this every time
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.imshow(self.manager.voxel_map.observations[0].instance)
        plt.show()

    def show_instance(self, instance: Instance, title: Optional[str] = None):
        """Show the instance in the voxel grid."""
        import matplotlib

        matplotlib.use("TkAgg")

        import matplotlib.pyplot as plt

        view = instance.get_best_view()
        plt.imshow(view.get_image())
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()
