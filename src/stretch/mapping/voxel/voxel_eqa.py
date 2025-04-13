# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import retry
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torch import Tensor

from stretch.llms.openai_client import OpenaiClient
from stretch.llms.prompts.eqa_prompt import (  # IMAGE_DESCRIPTION_PROMPT,
    EQA_PROMPT,
    EQA_SYSTEM_PROMPT_NEGATIVE,
    EQA_SYSTEM_PROMPT_POSITIVE,
)
from stretch.llms.qwen_client import Qwen25VLClient
from stretch.mapping.voxel.voxel_map_dynamem import SparseVoxelMap
from stretch.utils.voxel import scatter3d


class SparseVoxelMapEQA(SparseVoxelMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # To avoid using too much GPT, we use Qwen2.5-3b-vl-instruct-awq for image description.
        # self.image_description_client = Qwen25VLClient(prompt = None, model_size = "3B",)
        # self.image_description_client = OpenaiClient(
        #     prompt=IMAGE_DESCRIPTION_PROMPT, model="gpt-4o-mini"
        # )
        self.image_description_client = Qwen25VLClient(
            model_size="3B", quantization="int4", max_tokens=20
        )

        self.image_descriptions: List[List[str]] = []

        self.history_outputs: List[str] = []

        self.eqa_gpt_client = OpenaiClient(EQA_PROMPT, model="gpt-4o-2024-05-13")

        self.positive_score_client = OpenaiClient(EQA_SYSTEM_PROMPT_POSITIVE, model="gpt-4o")

        self.negative_score_client = OpenaiClient(EQA_SYSTEM_PROMPT_NEGATIVE, model="gpt-4o")

        # self.positive_score_client = Qwen25Client(EQA_SYSTEM_PROMPT_POSITIVE, model_type = "Deepseek", model_size = "1.5B")

        # self.negative_score_client = Qwen25Client(EQA_SYSTEM_PROMPT_NEGATIVE, model_type = "Deepseek", model_size = "1.5B")

    def query_answer(self, question: str, relevant_objects: List[str]):
        messages: List[Dict[str, Any]] = [{"type": "text", "text": "Question: " + question}]
        messages.append({"type": "text", "text": "HISTORY: "})
        for (i, history_output) in enumerate(self.history_outputs):
            messages.append({"type": "text", "text": "Iteration_" + str(i) + ":" + history_output})
        # messages.append({"role": "user", "content": [{"type": "input_text", "text": question}]})
        img_idx = 0

        # Log the text input and image input
        if not os.path.exists(self.log + "/" + str(len(self.image_descriptions))):
            os.makedirs(self.log + "/" + str(len(self.image_descriptions)))
            input_texts = ""
            for message in messages:
                input_texts += message["text"] + "\n"
            with open(
                self.log + "/" + str(len(self.image_descriptions)) + "/input.txt", "w"
            ) as file:
                file.write(input_texts)

        all_obs_ids = set()

        for relevant_object in relevant_objects:
            # Limit the total number of images to 5
            image_ids, _, _ = self.find_all_images(
                relevant_object,
                min_similarity_threshold=0.1,
                max_img_num=5 // len(relevant_objects),
                min_point_num=40,
            )
            for obs_id in image_ids:
                obs_id = int(obs_id) - 1
                all_obs_ids.add(obs_id)

        for obs_id in all_obs_ids:
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Save the input images
            image.save(
                self.log + "/" + str(len(self.image_descriptions)) + "/" + str(img_idx) + ".jpg"
            )
            img_idx += 1

            # Transform the images into base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {  # type:ignore
                        "url": f"data:image/png;base64,{base64_encoded}",
                        "detail": "high",
                    },
                }
            )

        # Extract answers
        answer_outputs = (
            self.eqa_gpt_client(messages).replace("*", "").replace("/", "").replace("#", "").lower()
        )

        # Log LLM output
        with open(self.log + "/" + str(len(self.image_descriptions)) + "/output.txt", "w") as file:
            file.write(answer_outputs)

        # Answer outputs in the format "Caption: Reasoning: Answer: Confidence: Confidence_reasoning:"
        reasoning = (
            answer_outputs.split("reasoning:")[-1]
            .split("answer:")[0]
            .replace("\n", "")
            .replace("\t", "")
        )
        answer = (
            answer_outputs.split("answer:")[-1]
            .split("confidence:")[0]
            .replace("\n", "")
            .replace("\t", "")
        )
        confidence = "true" in answer_outputs.split("confidence:")[-1].split(
            "confidence_reasoning:"
        )[0].replace(" ", "").replace("\n", "").replace("\t", "")
        confidence_reasoning = (
            answer_outputs.split("confidence_reasoning:")[-1].replace("\n", "").replace("\t", "")
        )

        self.history_outputs.append(
            "Answer:"
            + answer
            + "\nReasoning:"
            + reasoning
            + "\nConfidence:"
            + str(confidence)
            + "\nReasoning:"
            + confidence_reasoning
        )

        # Debug answer and confidence
        print("Answer:", answer)
        print("Confidence:", confidence)
        print("Answer outputs:", answer_outputs)

        return reasoning, answer, confidence, confidence_reasoning

    def get_2d_alignment_heuristics_mllm(self, task: str):
        """
        Transform the similarity with text into a 2D value map that can be used to evaluate
        how much exploring to one point can benefit open vocabulary navigation

        Similarity is computed by mLLM
        """
        if self.voxel_pcd._points is None:
            return None

        # Extract image id for each 2d grid points
        obs_ids = self.voxel_pcd._obs_counts
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        max_height = int(self.obs_max_height / self.grid_resolution)
        grid_size = self.grid_size + [max_height]
        obs_ids = obs_ids[:, None]

        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")
        history = torch.max(history_ids, dim=-1).values
        history = torch.from_numpy(maximum_filter(history.float().numpy(), size=7))
        history[0:35, :] = history.max().item()
        history[-35:, :] = history.max().item()
        history[:, 0:35] = history.max().item()
        history[:, -35:] = history.max().item()

        selected_images = torch.unique(history).int()
        # history image id is 1-indexed, so we need to subtract 1 from scores
        image_descriptions = [
            self.image_descriptions[selected_image.item() - 1] for selected_image in selected_images
        ]
        scores = self.compute_image_heuristic(task, image_descriptions=image_descriptions)
        final_scores = torch.zeros(len(self.image_descriptions))
        # history image id is 1-indexed, so we need to subtract 1 from scores
        final_scores[selected_images - 1] = torch.Tensor(scores)
        # for s, d in zip(final_scores, self.image_descriptions):
        #     print(s.item(), d)
        # history image id is 1-indexed, so we need to subtract 1 from scores
        return torch.Tensor(final_scores[history.int() - 1])

    def compute_image_heuristic(
        self,
        task,
        image_descriptions: Optional[List[List[str]]] = None,
        num_samples=5,
        positive_weight=0.2,
        negative_weight=0.1,
    ):
        """
        Compute an exploration heuristic score to determine how valuable it is to explore the contents inside the image.

        Args:
            task: things you want the robot to do, e.g. "find a blue bottle", "Answer the question 'What is the color of the washing machine'"
            positive_weight / negative_weight: scores = positive_weight * positive_scores + negative_weight * negative_scores
        """
        if image_descriptions is None:
            image_descriptions = self.image_descriptions
        try:
            positive_scores, _ = self.score_images(
                task, num_samples=num_samples, positive=True, image_descriptions=image_descriptions
            )
            negative_scores, _ = self.score_images(
                task, num_samples=num_samples, positive=False, image_descriptions=image_descriptions
            )
        except Exception as excptn:
            positive_scores, negative_scores = {}, {}
            print("GPTs failed:", excptn)
        language_scores = [0] * len(image_descriptions)
        # key - 1 because the image id output by GPT is 1-indexed
        for key, value in positive_scores.items():
            language_scores[key - 1] += value * positive_weight
        for key, value in negative_scores.items():
            language_scores[key - 1] -= value * negative_weight
        # for i, language_score in enumerate(language_scores):
        #     print(i, language_score)
        return language_scores

    @retry.retry(tries=5)
    def score_images(
        self,
        task: str,
        num_samples=5,
        positive=True,
        image_descriptions: Optional[List[List[str]]] = None,
        verbose: bool = True,
    ):

        options = ""

        if image_descriptions is None:
            image_descriptions = self.image_descriptions

        if verbose:
            print(
                "Querying",
                len(image_descriptions),
                "images.",
                len(self.image_descriptions),
                "images in total.",
            )

        if len(image_descriptions) > 0:
            for i, cluster in enumerate(image_descriptions):
                cluser_string = ""
                for ob in cluster:
                    cluser_string += ob + ", "
                options += f"{i+1}. {cluser_string[:-2]}\n"

        if positive:
            messages = [
                {
                    "type": "text",
                    "text": f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I search next if I try to {task}?",
                }
            ]
            choices = self.positive_score_client.sample(messages, n_samples=num_samples)
        else:
            messages = [
                {
                    "type": "text",
                    "text": f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I avoid spending time searching if I try to {task}?",
                }
            ]
            choices = self.negative_score_client.sample(messages, n_samples=num_samples)

        answers = []
        reasonings = []
        for choice in choices:
            try:
                complete_response = choice.message.content
                # Make the response all lowercase
                complete_response = complete_response.lower()
                reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                if len(complete_response.split("answer:")) > 1:
                    answer = complete_response.split("answer:")[1].split("\n")[0]
                    # Separate the answers by commas
                    answers.append([int(x) for x in answer.split(",")])
                else:
                    answers.append([])
                reasonings.append(reasoning)
            except:
                answers.append([])

        # Flatten answers
        flattened_answers = [item for sublist in answers for item in sublist]
        # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
        filtered_flattened_answers = [
            x for x in flattened_answers if x >= 1 and x <= len(image_descriptions)
        ]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {
            x: filtered_flattened_answers.count(x) / len(answers)
            for x in set(filtered_flattened_answers)
        }

        if verbose:
            print("Task:", task, "Score type:", positive)
            for image_id in answer_counts.keys():
                print("Image_id:", image_id, "scores:", answer_counts[image_id])
                print("Image descriptions:", image_descriptions[image_id - 1])

        return answer_counts, reasonings

    def process_rgbd_images(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray
    ):
        """
        Process rgbd images for EQA. Besides building semantic memory, for each image, we also name the object name in the image for exploration purpose.
        """
        # Build semantic memory with images following the same process as Dynamem.
        super().process_rgbd_images(rgb, depth, intrinsics, pose)

        self.list_objects_in_an_image(rgb)

    def list_objects_in_an_image(
        self, image: Union[torch.Tensor, Image.Image, np.ndarray], max_tries: int = 3
    ):
        """
        Process each image to name the object in the image for exploration purpose.
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)
        # buffered = BytesIO()
        # pil_image.save(buffered, format="PNG")
        # img_bytes = buffered.getvalue()
        # base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
        # messages = [
        #     {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/png;base64,{base64_encoded}",
        #             "detail": "low",
        #         },
        #     }
        # ]

        prompt = "List representative objects in the image. Limit your answer in 10 words. E.G.: a table,chairs,doors"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # self.obs_count inherited from voxel_dynamem
        objects = []
        for _ in range(max_tries):
            try:
                object_names = self.image_description_client(messages)
                objects = object_names.split(",")[:5]
            except:
                objects = []
                continue
            else:
                break

        if len(objects) == 0:
            self.image_descriptions.append(["object"])
        else:
            self.image_descriptions.append(objects)

        print(objects)
