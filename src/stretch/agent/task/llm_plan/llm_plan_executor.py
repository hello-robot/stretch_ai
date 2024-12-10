# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task
from stretch.utils.llm_plan_wrapper import LLMPlanWrapper


class LLMPlanExecutor:
    def __init__(self, agent: RobotAgent, llm_plan: Optional[str] = None):
        # Sync these things
        self.agent = agent
        self.robot = agent.robot
        self.voxel_map = agent.get_voxel_map()
        self.navigation_space = agent.space
        self.semantic_sensor = agent.semantic_sensor
        self.parameters = agent.parameters
        self.instance_memory = agent.get_voxel_map().instances

        # Task information
        self.llm_plan = llm_plan
        self.task = None

        if llm_plan is not None:
            self.llm_plan_wrapper = LLMPlanWrapper(agent, self.llm_plan)

    def configure(self, llm_plan: Optional[str] = None):
        """Configure the task given a LLM plan."""
        self.llm_plan = llm_plan

    def run(self) -> Task:
        if self.llm_plan_wrapper is None:
            self.llm_plan_wrapper = LLMPlanWrapper(self.agent, self.llm_plan)
            # print("Compiling LLM plan...")

        self.llm_plan_wrapper.run()

        # return self.task
