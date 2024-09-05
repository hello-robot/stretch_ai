# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Operation, Task


class EmoteTask:
    """
    Creates a task queue with a given emote operation.
    """

    def __init__(self, agent: RobotAgent):
        super().__init__(agent)

        # random stuff that has to be synced...
        self.navigation_space = agent.space
        self.parameters = agent.parameters
        self.robot = agent.robot

    def get_task(self, emote_operation: Operation) -> Task:
        task = Task()
        task.add_operation(emote_operation)
        return task
