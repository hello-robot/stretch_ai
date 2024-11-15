# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Union

from stretch.agent.operations import (
    AvertGazeOperation,
    NodHeadOperation,
    ShakeHeadOperation,
    WaveOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Operation, Task


class EmoteTask:
    """
    Creates a task queue with a given emote operation.
    """

    def __init__(self, agent: RobotAgent):
        super().__init__()

        # random stuff that has to be synced...
        self.agent = agent
        self.navigation_space = agent.space
        self.parameters = agent.parameters
        self.robot = agent.robot

    def get_task(self, emote_operation: Union[Operation, str]) -> Task:
        task = Task()
        if isinstance(emote_operation, Operation):
            task.add_operation(emote_operation)
        elif isinstance(emote_operation, str):
            if emote_operation == "nod" or emote_operation == "nod_head":
                task.add_operation(NodHeadOperation("emote", self.agent))
            elif emote_operation == "shake" or emote_operation == "shake_head":
                task.add_operation(ShakeHeadOperation("emote", self.agent))
            elif emote_operation == "wave":
                task.add_operation(WaveOperation("emote", self.agent))
            elif emote_operation == "avert" or emote_operation == "avert_gaze":
                task.add_operation(AvertGazeOperation("emote", self.agent))
            else:
                raise ValueError(f"Invalid emote operation: {emote_operation}")
        else:
            raise TypeError(f"Expected Operation or str, got {type(emote_operation)}")
        return task
