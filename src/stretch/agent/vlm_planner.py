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

from stretch.agent.robot_agent import RobotAgent
from stretch.llms.multi_crop_openai_client import MultiCropOpenAIClient


class VLMPlanner:
    def __init__(self, agent: RobotAgent, api_key: Optional[str] = None) -> None:
        """This is a connection to a VLM for getting a plan based on language commands.

        Args:
            agent (RobotAgent): the agent
            api_key (str): the API key for the VLM. Optional; if not provided, will be read from the environment variable OPENAI_API_KEY. If not found there, will prompt the user for it.
        """

        self.agent = agent

        # Load parameters file from the agent
        self.parameters = agent.parameters
        self.voxel_map = agent.voxel_map

        # TODO: put these into config
        img_size = 256
        temperature = 0.2
        max_tokens = 50
        with open(
            "src/stretch/llms/prompts/obj_centric_vlm.txt",
            "r",
        ) as f:
            prompt = f.read()

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("You are using GPT4v for planning, please type in your openai key: ")
        self.api_key = api_key

        self.gpt_agent = MultiCropOpenAIClient(
            cfg=dict(
                img_size=img_size,
                prompt=prompt,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self.gpt_agent.reset()

    def set_agent(self, agent: RobotAgent) -> None:
        """Set the agent for the VLM planner.

        Args:
            agent (RobotAgent): the agent
        """
        self.agent = agent

    def plan(
        self,
        current_pose=None,
        show_prompts=False,
        show_plan=False,
        plan_file: str = "vlm_plan.txt",
        query=None,
        plan_with_reachable_instances=False,
        plan_with_scene_graph=False,
    ) -> str:
        """This is a connection to a VLM for getting a plan based on language commands.

        Args:
            current_pose(np.ndarray): the current pose of the robot
            show_prompts(bool): whether to show prompts
            show_plan(bool): whether to show the plan
            plan_file(str): the name of the file to save the plan to
            query(str): the query to send to the VLM

        Returns:
            str: the plan
        """
        world_representation = self.agent.get_object_centric_observations(
            task=query,
            current_pose=current_pose,
            show_prompts=show_prompts,
            plan_with_reachable_instances=plan_with_reachable_instances,
            plan_with_scene_graph=plan_with_scene_graph,
        )
        output = self.get_output_from_gpt(world_representation, task=query)
        if show_plan:
            import re

            import matplotlib.pyplot as plt

            if output == "explore":
                print(">>>>>> Planner cannot find a plan, the robot should explore more >>>>>>>>>")
            elif output == "gpt API error":
                print(">>>>>> there is something wrong with the planner api >>>>>>>>>")
            else:
                actions = output.split("; ")
                plt.clf()
                for action_id, action in enumerate(actions):
                    crop_id = int(re.search(r"img_(\d+)", action).group(1))
                    global_id = world_representation.object_images[crop_id].instance_id
                    plt.subplot(1, len(actions), action_id + 1)
                    plt.imshow(
                        self.voxel_map.get_instances()[global_id].get_best_view().get_image()
                    )
                    plt.title(action.split("(")[0] + f" instance {global_id}")
                    plt.axis("off")
                plt.suptitle(f"Task: {query}")
                plt.show()
                plt.savefig("plan.png")

        if self.parameters.get("save_vlm_plan", True):
            with open(plan_file, "w") as f:
                f.write(output)
            print(f"Task plan generated from VLMs has been written to {plan_file}")
        return actions, world_representation

    def get_output_from_gpt(self, world_rep, task: str):

        plan = self.gpt_agent.act_on_observations(world_rep, goal=task, debug_path=None)
        return plan
