# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
from io import BytesIO
from typing import List, Union

import numpy as np
import retry
import torch
from PIL import Image
from torch import Tensor

from stretch.llms.openai_client import OpenaiClient

# from stretch.llms.qwen_client import Qwen25VLClient
from stretch.llms.prompts.eqa_prompt import (
    EQA_PROMPT,
    EQA_SYSTEM_PROMPT_NEGATIVE,
    EQA_SYSTEM_PROMPT_POSITIVE,
    IMAGE_DESCRIPTION_PROMPT,
)
from stretch.mapping.voxel.voxel_map_dynamem import SparseVoxelMap


class SparseVoxelMapEQA(SparseVoxelMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # To avoid using too much GPT, we use Qwen2.5-3b-vl-instruct-awq for image description.
        # self.image_description_client = Qwen25VLClient(prompt = None, model_size = "3B",)
        self.image_description_client = OpenaiClient(
            prompt=IMAGE_DESCRIPTION_PROMPT, model="gpt-4o-mini"
        )

        self.image_descriptions: List[List[str]] = {}

        self.history_outputs: List[str] = []

        self.eqa_gpt_client = OpenaiClient(EQA_PROMPT, model="gpt-4o")

        self.positive_score_client = OpenaiClient(EQA_SYSTEM_PROMPT_POSITIVE, model="gpt-4o-mini")

        self.negative_score_client = OpenaiClient(EQA_SYSTEM_PROMPT_NEGATIVE, model="gpt-4o-mini")

    def query_answer(self, question: str, relevant_objects: List[str]):
        messages = [{"role": "user", "content": question}]
        for (i, history_output) in enumerate(self.history_outputs):
            messages.append({"role": "assistant", "content": history_output})
        messages.append({"role": "user", "content": "Question: " + question})
        for relevant_object in relevant_objects:
            image_ids, _, _ = self.find_all_images(relevant_object)
            for obs_id in image_ids:
                obs_id = int(obs_id) - 1
                rgb = np.copy(self.observations[obs_id].rgb.numpy())
                image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                messages.append(
                    {
                        "type": "image_url",
                        "image_url": {  # type: ignore
                            "url": f"data:image/png;base64,{base64_encoded}",
                            "detail": "low",
                        },
                    }
                )
        answer = self.eqa_gpt_client(messages)
        print("=" * 30)
        print(answer)
        self.history_outputs.append(answer)

    def get_2d_alignment_heuristics_mllm(self, task: str, debug: bool = False):
        """
        Transform the similarity with text into a 2D value map that can be used to evaluate
        how much exploring to one point can benefit open vocabulary navigation

        Similarity is computed by mLLM
        """
        if self.voxel_pcd._points is None:
            return None
        scores = np.array(self.compute_image_heuristic(task))
        _, _, history = self.get_2d_map(return_history_id=True)
        return torch.Tensor(scores[np.array(history)])

    def compute_image_heuristic(
        self, task, num_samples=5, positive_weight=2.0, negative_weight=1.0
    ):
        try:
            positive_scores, _ = self.score_images(task, num_samples=num_samples, positive=True)
            negative_scores, _ = self.score_images(task, num_samples=num_samples, positive=False)
        except Exception as excptn:
            positive_scores, negative_scores = {}, {}
            print("GPTs failed:", excptn)
        language_scores = [0] * len(self.image_descriptions)
        for key, value in positive_scores.items():
            language_scores[key] += value * positive_weight
        for key, value in negative_scores.items():
            language_scores[key] -= value * negative_weight
        return language_scores

    @retry.retry(tries=5)
    def score_images(self, task, num_samples=5, positive=True):

        if len(self.image_descriptions) > 0:
            options = ""
            for i, cluster in enumerate(self.image_descriptions):
                cluser_string = ""
                for ob in cluster:
                    cluser_string += ob + ", "
                options += f"{i+1}. {cluser_string[:-2]}\n"

        if positive:
            messages = [
                {
                    "type": "text",
                    "content": f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I search next if I try to {task}?",
                }
            ]
            choices = self.positive_score_client.sample(messages, n_samples=num_samples)
        else:
            messages = [
                {
                    "type": "text",
                    "content": f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I avoid spending time searching if I try to {task}?",
                }
            ]
            choices = self.negative_score_client.sample(messages, n_samples=num_samples)

        answers = []
        reasonings = []
        for choice in choices:
            try:
                complete_response = choice.message["content"]
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
            x for x in flattened_answers if x >= 1 and x <= len(self.image_descriptions)
        ]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {
            x: filtered_flattened_answers.count(x) / len(answers)
            for x in set(filtered_flattened_answers)
        }

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
        prompt = "List as many objects as possible in the image, but limit your answer in 10 objects. You can repeat the object name if it appears for multiple times. E.G. a yellow banana,a purple bottle,a table,a chair"
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
        messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_encoded}",
                    "detail": "low",
                },
            },
            {"type": "text", "text": prompt},
        ]

        # self.obs_count inherited from voxel_dynamem
        objects = []
        for _ in range(max_tries):
            try:
                object_names = self.image_description_client(messages)
                objects = object_names.split(",")
            except:
                objects = []
                continue
            else:
                break

        if len(objects) == 0:
            self.image_descriptions.append(["object"])
        else:
            self.image_descriptions.append(objects)
