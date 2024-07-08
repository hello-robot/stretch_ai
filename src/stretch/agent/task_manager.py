import abc
from typing import Optional, Union

from stretch.mapping.instance import Instance

from .robot_agent import RobotAgent


class TaskManager(abc.ABC):
    """A Task Manager is a blackboard managing the full task state, with some helper functions to make it easier to plan and reason about what to do next. It should be implemented for the particular task at hand. Part of its job is to create the Task object holding the actual task instance."""

    def __init__(self, agent: RobotAgent):
        self.agent = agent
        self.reset_object_plans()

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
        print(instance_id, self.unreachable_instances)
        return instance_id in self.unreachable_instances

    @abc.abstractmethod
    def get_task(self, *args, **kwargs):
        raise NotImplementedError
