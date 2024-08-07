# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
# 
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


import abc
from typing import Optional, Union

from stretch.agent.robot_agent import RobotAgent
from stretch.mapping.instance import Instance


class TaskManager(abc.ABC):
    """A Task Manager is a blackboard managing the full task state, with some helper functions to make it easier to plan and reason about what to do next. It should be implemented for the particular task at hand. Part of its job is to create the Task object holding the actual task instance."""

    def __init__(self, agent: RobotAgent):
        self.agent = agent
        self.reset_object_plans()

        # Sync these things
        self.robot = agent.robot
        self.voxel_map = agent.voxel_map
        self.navigation_space = agent.space
        self.semantic_sensor = agent.semantic_sensor
        self.parameters = agent.parameters
        self.instance_memory = agent.voxel_map.instances

        if agent.voxel_map.use_instance_memory:
            assert (
                self.instance_memory is not None
            ), "Make sure instance memory was created! This is configured in parameters file."

    def reset_object_plans(self):
        """Clear stored object planning information."""
        self.plans = {}
        self.unreachable_instances = set()

    def set_instance_as_unreachable(self, instance: Union[int, Instance]) -> None:
        """Mark an instance as unreachable."""
        if isinstance(instance, Instance):
            instance_id = instance.id
        elif isinstance(instance, int):
            instance_id = instance
        else:
            raise ValueError("Instance must be an Instance object or an int")
        self.unreachable_instances.add(instance_id)

    def is_instance_unreachable(self, instance: Union[int, Instance]) -> bool:
        """Check if an instance is unreachable."""
        if isinstance(instance, Instance):
            instance_id = instance.id
        elif isinstance(instance, int):
            instance_id = instance
        else:
            raise ValueError("Instance must be an Instance object or an int")
        return instance_id in self.unreachable_instances

    @abc.abstractmethod
    def get_task(self, *args, **kwargs):
        raise NotImplementedError
