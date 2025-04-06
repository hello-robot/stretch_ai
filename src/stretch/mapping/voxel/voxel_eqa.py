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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import retry
import torch
from PIL import Image
from torch import Tensor

from stretch.llms.openai_client import OpenaiClient

# from stretch.llms.qwen_client import Qwen25VLClient, Qwen25Client
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
        # self.image_description_client = Qwen25VLClient(
        #     prompt=None, model_size="3B", quantization = "awq"
        # )

        self.image_descriptions: List[List[str]] = []

        self.history_outputs: List[str] = []

        self.eqa_gpt_client = OpenaiClient(EQA_PROMPT, model="gpt-4o")

        self.positive_score_client = OpenaiClient(EQA_SYSTEM_PROMPT_POSITIVE, model="gpt-4o")

        self.negative_score_client = OpenaiClient(EQA_SYSTEM_PROMPT_NEGATIVE, model="gpt-4o")

        # self.positive_score_client = Qwen25Client(EQA_SYSTEM_PROMPT_POSITIVE, model_type = "Deepseek", model_size = "1.5B")

        # self.negative_score_client = Qwen25Client(EQA_SYSTEM_PROMPT_NEGATIVE, model_type = "Deepseek", model_size = "1.5B")

    def query_answer(self, question: str, relevant_objects: List[str]):
        messages: List[Dict[str, Any]] = [{"type": "text", "text": question}]
        messages.append({"type": "text", "text": "HISTORY"})
        for (i, history_output) in enumerate(self.history_outputs):
            messages.append({"type": "text", "text": history_output})
        # messages.append({"role": "user", "content": [{"type": "input_text", "text": question}]})
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
                        "image_url": {  # type:ignore
                            "url": f"data:image/png;base64,{base64_encoded}",
                            "detail": "low",
                        },
                    }
                )
        answer_outputs = self.eqa_gpt_client(messages)
        # Answer outputs in the format "Caption: Reasoning: Answer: Confidence: Confidence_reasoning:"
        reasoning = answer_outputs.split("Reasoning:")[-1].split("Answer:")[0]
        answer = answer_outputs.split("Answer:")[-1].split("Confidence:")[0]
        confidence = (
            answer_outputs.split("Confidence:")[-1]
            .split("Confidence_reasoning:")[0]
            .replace(" ", "")
            == "True"
        )
        confidence_reasoning = answer_outputs.split("Confidence_reasoning:")[-1]

        print("Answer:", answer)
        print("Confidence:", confidence)

        return reasoning, answer, confidence, confidence_reasoning

    def get_2d_alignment_heuristics_mllm(self, task: str):
        """
        Transform the similarity with text into a 2D value map that can be used to evaluate
        how much exploring to one point can benefit open vocabulary navigation

        Similarity is computed by mLLM
        """
        if self.voxel_pcd._points is None:
            return None
        _, _, history = self.get_2d_map(return_history_id=True)
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
        positive_weight=0.4,
        negative_weight=0.2,
    ):
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
        verbose: bool = False,
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
            print(answer_counts, image_descriptions, answers, reasonings)
            for a in answer_counts.keys():
                print(a, answer_counts[a], image_descriptions[a - 1])

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
            # {"type": "text", "text": prompt},
        ]

        # prompt = "List as many objects as possible in the image, but limit your answer in 5 objects. E.G. a yellow banana,some purple bottles,a table,3 chairs"
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": pil_image,
        #             },
        #             {
        #                 "type": "text",
        #                 "image": prompt,
        #             },
        #         ]
        #     }
        # ]

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

        print(objects)
