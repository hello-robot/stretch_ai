# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64

# import json
import os
import time
from enum import Enum

# from io import BytesIO
from typing import List, Union

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

if "OPENAI_API_KEY" in os.environ:
    client = OpenAI()
else:
    print("GPT token has not been set up yet!")


def get_latest_image(output_folder):
    png_files = [file for file in os.listdir(output_folder) if file.startswith("current_img_")]
    indices = [int(f.split("current_img_")[1].split(".png")[0]) for f in png_files]
    return output_folder + f"/current_img_{np.max(indices)}.png"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_planner_response(
    frontier_node_list, room_node_list, region_node_list, object_node_list, use_image=True
):
    class Goto_frontier_node_step(BaseModel):
        explanation_frontier: str
        frontier_id: frontier_node_list

    class Goto_object_node_step(BaseModel):
        explanation_room: str
        # explanation_region: str
        explanation_obj: str
        room_id: room_node_list
        # region_id: region_node_list
        object_id: object_node_list

    class Answer(BaseModel):
        explanation_ans: str
        answer: str
        explanation_conf: str
        confidence_level: float
        is_confident: bool

    class PlannerResponse(BaseModel):
        steps: List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        answer: Answer
        image_description: str
        scene_graph_description: str

    class PlannerResponseNoFrontiers(BaseModel):
        steps: List[Goto_object_node_step]
        answer: Answer
        image_description: str
        scene_graph_description: str

    class PlannerResponseNoImage(BaseModel):
        steps: List[Union[Goto_object_node_step, Goto_frontier_node_step]]
        answer: Answer
        scene_graph_description: str

    class PlannerResponseNoFrontiersNoImage(BaseModel):
        steps: List[Goto_object_node_step]
        answer: Answer
        scene_graph_description: str

    if use_image:
        if frontier_node_list is None:
            return PlannerResponseNoFrontiers
        else:
            return PlannerResponse
    else:
        if frontier_node_list is None:
            return PlannerResponseNoFrontiersNoImage
        else:
            return PlannerResponseNoImage


class VLMPLannerEQAGPT:
    def __init__(self, vlm_type, sg_sim, question, output_path, use_image=True, add_history=True):

        self._question = question
        self._output_path = output_path
        self._vlm_type = vlm_type
        self._use_image = use_image

        self._history = ""
        self.full_plan = ""
        self._t = 0
        self._add_history = add_history

        self._outputs_to_save = [f"Question: {self._question}.\n"]
        self.sg_sim = sg_sim

    @property
    def t(self):
        return self._t

    def get_actions(self):
        object_node_list = Enum(
            "object_node_list",
            {
                id: name
                for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)
            },
            type=str,
        )
        if len(self.sg_sim.frontier_node_ids) > 0:
            frontier_node_list = Enum(
                "frontier_node_list", {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str
            )
        else:
            # frontier_node_list = Enum('frontier_node_list', {'frontier_0': 'Do not choose this option. No more frontiers left.'}, type=str)
            frontier_node_list = None

        room_node_list = Enum(
            "room_node_list",
            {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)},
            type=str,
        )
        region_node_list = Enum(
            "region_node_list", {ac: ac for ac in self.sg_sim.region_node_ids}, type=str
        )
        return frontier_node_list, room_node_list, region_node_list, object_node_list

    @property
    def agent_role_prompt(self):
        scene_graph_desc = "A scene graph represents an indoor environment in node list.\n \
         Each node can represent an object, an unexplored frontier, or the agent current position \n"
        current_state_des = "'CURRENT AGENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location. "

        if self._use_image:
            current_state_des += "Additionally, you will also be given the current view of the agent as an image.\n \
                The information in the image is very useful and should be prioritized when answering the question! \n"

        prompt = f"""You are an excellent hierarchical graph planning agent. 
            Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
            {scene_graph_desc}. {current_state_des} 
            Given the current state information, try to answer the question. 
            Explain the reasoning for selecting the answer.
            Finally, report whether you are confident in answering the question. 
            Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
            Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
            Do not use just commensense knowledge to decide confidence. 
            Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. 
            Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current environment. 
            Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
            If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of steps in the environment: Goto_object_node_step or Goto_frontier_node_step 
            You also have to choose the next action, one which will enable you to answer the question better. 
            Goto_object_node_step: Navigates near a certain object in the scene graph. Choose this action to get a good view of the region around this object, if you think going near this object will help you answer the question better.
            Important to note, the scene contains incomplete information about the environment (objects maybe missing, relationships might be unclear), so it is useful to go near relevant objects to get a better view to answer the question. 
            Use a scene graph as an imperfect guide to lead you to relevant regions to inspect.
            Choose the object in a hierarchical manner by first reasoning about which room you should goto to best answer the question, and then choose the specific object. \n
            Goto_frontier_node_step: If you think that using action "Goto_object_node_step" is not useful, in other words, if you think that going near any of the object nodes in the current scene graph will not provide you with any useful information to answer the question better, then choose this action.
            This action will navigate you to a frontier (unexplored) region of the environment and will provide you information about new objects/rooms not yet in the scene graph. It will expand the scene graph. 
            Choose this frontier based on the objects connected this frontier, in other words, Goto the frontier near which you see objects that are useful for answering the question or seem useful as a good exploration direction. Explain reasoning for choosing this frontier, by listing the list of objects (<id> and <name>) connected to this frontier node via a link (refer to scene graph) \n \
            
            While choosing either of the above actions, play close attention to 'HISTORY' especially the previous 'Action's to see if you have taken the same action at previous timesteps. 
            Avoid taking the same actions you have taken before.
            Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions.
            Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.
            """
        # prompt += "You should go near the blue couch before answering the question with confidence. You should see a full image of the couch before answering with confidence"
        prompt_no_image = f"""You are an excellent hierarchical graph planning agent. 
            Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
            {scene_graph_desc}. {current_state_des} 
            Given the current state information, try to answer the question. Explain the reasoning for selecting the answer.
            Finally, report whether you are confident in answering the question. 
            Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
            Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
            Do not use just commensense knowledge to decide confidence. 
            Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. 
            Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current environment. 
            Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
            If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of steps in the environment: Goto_object_node_step or Goto_frontier_node_step 
            You also have to choose the next action, one which will enable you to answer the question better. 
            Goto_object_node_step: Navigates near a certain object in the scene graph. Choose this action to go to the region around this object, if you think going near this object will help you answer the question better.
            Choose the object in a hierarchical manner by first reasoning about which room you should goto to best answer the question, and then choose the specific object. \n
            Goto_frontier_node_step: If you think that using action "Goto_object_node_step" is not useful, in other words, if you think that going near any of the object nodes in the current scene graph will not provide you with any useful information to answer the question better, then choose this action.
            This action will navigate you to a frontier (unexplored) region of the environment and will provide you information about new objects/rooms not yet in the scene graph. It will expand the scene graph. 
            Choose this frontier based on the objects connected this frontier, in other words, Goto the frontier near which there are objects useful for answering the question or seem useful as a good exploration direction. Explain reasoning for choosing this frontier, by listing the list of objects (<id> and <name>) connected to this frontier node via a link (refer to scene graph) \n \
            
            While choosing either of the above actions, play close attention to 'HISTORY' especially the previous 'Action's to see if you have taken the same action at previous timesteps. 
            Avoid taking the same actions you have taken before.
            Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.
            """

        if self._use_image:
            return prompt
        else:
            return prompt_no_image

    def get_current_state_prompt(self, scene_graph, agent_state):
        # TODO(saumya): Include history
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n  "

        if self._add_history:
            prompt += f"HISTORY: {self._history}"

        return prompt

    def update_history(self, agent_state, step, answer, target_pose):
        if step.__class__.__name__ == "Goto_object_node_step":
            action = f"Goto object_id:{step.object_id.name} object name: {step.object_id.value}"
        else:
            action = (
                f"Goto frontier_id:{step.frontier_id.name} frontier name: {step.frontier_id.value}"
            )

        # Answer(t={self.t}): {answer.answer.name} {answer.answer.value}
        last_step = f"""
            [Agent state(t={self.t}): {agent_state}, 
            Action(t={self.t}): {action}, 
            Answer(t={self.t}): {answer},
            Confidence(t={self.t}):  Confident: {answer.is_confident}, Confidence level:{answer.confidence_level}  \n
        """
        self._history += last_step

    def get_gpt_output(self, current_state_prompt):

        messages = [
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"QUESTION: {self._question}"},
            {"role": "user", "content": f"CURRENT STATE: {current_state_prompt}."},
        ]

        # images = self.sg_sim.images

        if self._use_image:

            base64_image = encode_image(get_latest_image(self._output_path))
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            )

            # for image in images:
            #     buffered = BytesIO()
            #     image.save(buffered, format="PNG")
            #     img_bytes = buffered.getvalue()
            #     base64_image = base64.b64encode(img_bytes).decode("utf-8")
            #     # base64_image = encode_image(get_latest_image(self._output_path))
            #     messages.append(
            #         {
            #             "role": "user",
            #             "content": [
            #                 {
            #                     "type": "text",
            #                     "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question.",
            #                 },
            #                 {
            #                     "type": "image_url",
            #                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            #                 },
            #             ],
            #         }
            #     )

        frontier_node_list, room_node_list, region_node_list, object_node_list = self.get_actions()

        succ = False
        while not succ:
            try:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    model=self._vlm_type,
                    messages=messages,
                    response_format=create_planner_response(
                        frontier_node_list,
                        room_node_list,
                        region_node_list,
                        object_node_list,
                        use_image=self._use_image,
                    ),
                )
                plan = completion.choices[0].message
                if not (
                    plan.refusal
                ):  # If the model refuses to respond, you will get a refusal message
                    succ = True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60s")
                # import ipdb; ipdb.set_trace()
                time.sleep(10)

        plan = completion.choices[0].message

        if len(plan.parsed.steps) > 0:
            step = plan.parsed.steps[0]
        else:
            step = None

        if self._use_image:
            img_desc = plan.parsed.image_description
        else:
            img_desc = " "

        return step, plan.parsed.answer, img_desc, plan.parsed.scene_graph_description

    def get_next_action(self):
        # self.sg_sim.update()

        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self.get_current_state_prompt(
            self.sg_sim.scene_graph_str, agent_state
        )

        sg_desc = ""
        step, answer, img_desc, sg_desc = self.get_gpt_output(current_state_prompt)

        # Saving outputs to file
        self._outputs_to_save.append(
            f"At t={self._t}: \n \
                                        Agent state: {agent_state} \n \
                                        LLM output: {step}. \n \
                                        Answer: {answer} \n \
                                        Image desc: {img_desc} \n \
                                        Scene graph desc: {sg_desc} \n \n"
        )
        self.full_plan = " ".join(self._outputs_to_save)
        with open(self._output_path + "/llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        print(f"At t={self._t}: \n {step} \n {answer}")

        if step is None:
            return (
                None,
                None,
                answer.is_confident,
                answer.confidence_level,
                answer.answer,
                answer.explanation_ans,
                # get_latest_image(self._output_path),
                self.sg_sim.images,
            )

        if step.__class__.__name__ == "Goto_object_node_step":
            target_pose = self.sg_sim.get_position_from_id(step.object_id.name)
            target_id = step.object_id.name
        else:
            target_pose = self.sg_sim.get_position_from_id(step.frontier_id.name)
            target_id = step.frontier_id.name

        if self._add_history:
            self.update_history(agent_state, step, answer, target_pose)

        self._t += 1
        return (
            target_pose,
            target_id,
            answer.is_confident,
            answer.confidence_level,
            answer.answer,
            answer.explanation_ans,
            # get_latest_image(self._output_path),
            self.sg_sim.images,
        )
