# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

from stretch.agent.base import TaskManager
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Operation, Task


class EmoteManager(TaskManager):
    """
    Provides a minimal interface with the TaskManager class.
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
