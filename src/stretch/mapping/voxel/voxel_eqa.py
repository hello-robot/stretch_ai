# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import retry
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torch import Tensor

# from stretch.llms.openai_client import OpenaiClient
from stretch.llms.gemini_client import GeminiClient
from stretch.llms.prompts.eqa_prompt import EQA_SYSTEM_PROMPT_NEGATIVE, EQA_SYSTEM_PROMPT_POSITIVE
from stretch.llms.qwen_client import Qwen25VLClient
from stretch.mapping.voxel.voxel_map_dynamem import SparseVoxelMap
from stretch.utils.voxel import scatter3d


class SparseVoxelMapEQA(SparseVoxelMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cached question
        self._question: Optional[str] = None
        self.relevant_objects: Optional[list] = None

        # To avoid using too much GPT, we use Qwen2.5-3b-vl-instruct for image description.
        self.image_description_client = Qwen25VLClient(
            model_size="3B", quantization="int4", max_tokens=20
        )

        self.image_descriptions: List[Tuple[List[str], List[int]]] = []

        self.history_outputs: List[str] = []

        self.positive_score_client = GeminiClient(
            EQA_SYSTEM_PROMPT_POSITIVE, model="gemini-2.0-flash-lite"
        )

        self.negative_score_client = GeminiClient(
            EQA_SYSTEM_PROMPT_NEGATIVE, model="gemini-2.0-flash-lite"
        )

    def extract_relevant_objects(self, question: str):
        if self._question != question:
            self._question = question
            # The cached question is not the same as the question provided
            prompt = """
                Assume there is an agent doing Question Answering in an environment.
                When it receives a question, you need to tell the agent few objects (preferably 1-3) it needs to pay special attention to.
                Example:
                    Where is the pen?
                    pen

                    Is there grey cloth on cloth hanger?
                    gery cloth,cloth hanger
            """
            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "text",
                            "text": self._question,
                        },
                    ],
                }
            ]
            # To avoid initializing too many clients and using up too much memory, I reused the client generating the image descriptions even though it is a VL model
            self.relevant_objects = self.image_description_client(messages).split(",")
            print("relevant objects to look at", self.relevant_objects)
            self.history_outputs = []

    def log_text(self, commands):
        """
        Log the text input and image input into some files for debugging and visualization
        """
        if not os.path.exists(self.log + "/" + str(len(self.image_descriptions))):
            os.makedirs(self.log + "/" + str(len(self.image_descriptions)))
            input_texts = ""
            for command in commands:
                input_texts += command + "\n"
            with open(
                self.log + "/" + str(len(self.image_descriptions)) + "/input.txt", "w"
            ) as file:
                file.write(input_texts)

    def parse_answer(self, answer_outputs: str):

        # Log LLM output
        with open(self.log + "/" + str(len(self.image_descriptions)) + "/output.txt", "w") as file:
            file.write(answer_outputs)

        # Answer outputs in the format "Caption: Reasoning: Answer: Confidence: Action: Confidence_reasoning:"
        def extract_between(text, start, end):
            try:
                return (
                    text.split(start, 1)[1]
                    .split(end, 1)[0]
                    .strip()
                    .replace("\n", "")
                    .replace("\t", "")
                )
            except IndexError:
                return ""

        def extract_after(text, start):
            try:
                return text.split(start, 1)[1].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        reasoning = extract_between(answer_outputs, "reasoning:", "answer:")
        answer = extract_between(answer_outputs, "answer:", "confidence:")
        confidence_text = extract_between(answer_outputs, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "")
        action = extract_between(answer_outputs, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(answer_outputs, "confidence_reasoning:")

        return reasoning, answer, confidence, action, confidence_reasoning

    def get_active_image_descriptions(self):
        """
        Return a list of image descriptions that are still active. By active it means there is still some voxel in voxel map associated with it.
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
        history = torch.from_numpy(maximum_filter(history.float().numpy(), size=5))
        history[0:35, :] = history.max().item()
        history[-35:, :] = history.max().item()
        history[:, 0:35] = history.max().item()
        history[:, -35:] = history.max().item()
        # from matplotlib import pyplot as plt
        # plt.imshow(history)
        # plt.show()

        selected_images = torch.unique(history).int()
        # history image id is 1-indexed, so we need to subtract 1 from scores
        return (
            history,
            selected_images,
            [
                self.image_descriptions[selected_image.item() - 1]
                for selected_image in selected_images
            ],
        )

    def get_2d_alignment_heuristics_mllm(self, task: str):
        """
        Transform the similarity with text into a 2D value map that can be used to evaluate
        how much exploring to one point can benefit open vocabulary navigation

        Similarity is computed by mLLM
        """
        history, selected_images, image_descriptions = self.get_active_image_descriptions()
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
        image_descriptions: Optional[List[Tuple[List[str], List[int]]]] = None,
        num_samples=4,
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
        return language_scores

    @retry.retry(tries=5)
    def score_images(
        self,
        task: str,
        num_samples=4,
        positive=True,
        image_descriptions: Optional[List[Tuple[List[str], List[int]]]] = None,
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
            for i, (cluster, _) in enumerate(image_descriptions):
                cluser_string = ""
                for ob in cluster:
                    cluser_string += ob + ", "
                options += f"{i+1}. {cluser_string[:-2]}\n"

        if positive:
            messages = f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I search next if I try to {task}?"
            choices = self.positive_score_client.sample(messages, n_samples=num_samples)
        else:
            messages = f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I avoid spending time searching if I try to {task}?"
            choices = self.negative_score_client.sample(messages, n_samples=num_samples)

        answers = []
        reasonings = []
        for choice in choices:
            try:
                complete_response = choice.lower()
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
        Extract visual clues (a list of featured objects) from the image observation and add the clues to a list
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)

        prompt = "List representative objects in the image (excluding floor and wall) Limit your answer in 10 words. E.G.: a table,chairs,doors"
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

        obs_ids = self.voxel_pcd._obs_counts
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        grid_coord = list(
            self.xy_to_grid_coords(
                torch.mean(xyz[obs_ids == obs_ids.max()], dim=0)[:2].int().cpu().numpy()
            )
        )
        for i in range(len(grid_coord)):
            grid_coord[i] = int(grid_coord[i])

        if len(objects) == 0:
            self.image_descriptions.append((["object"], grid_coord))
        else:
            self.image_descriptions.append((objects, grid_coord))

        print(objects)
