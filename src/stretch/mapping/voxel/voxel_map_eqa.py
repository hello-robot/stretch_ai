# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import numpy as np
import torch
from PIL import Image

from stretch.llms.gemini_client import GeminiClient
from stretch.llms.prompts.eqa_prompt import EQA_PROMPT

from .voxel import SparseVoxelMapProxy
from .voxel_eqa import SparseVoxelMapEQA
from .voxel_map_dynamem import SparseVoxelMapNavigationSpace as SparseVoxelMapNavigationSpaceBase


class SparseVoxelMapNavigationSpace(SparseVoxelMapNavigationSpaceBase):

    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        voxel_map: SparseVoxelMapEQA | SparseVoxelMapProxy,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
        alignment_heuristics_type: str = "mllm",
    ):
        """
        Parameters:
            alignment_heuristics_type: If we want to compute alignment heuristics for exploration, we can choose either using "encoder" or "mllm"
        """
        super().__init__(
            voxel_map=voxel_map,
            step_size=step_size,
            rotation_step_size=rotation_step_size,
            use_orientation=use_orientation,
            orientation_resolution=orientation_resolution,
            dilate_frontier_size=dilate_frontier_size,
            dilate_obstacle_size=dilate_obstacle_size,
            extend_mode=extend_mode,
        )
        # This assignment has already happened in extended class, but assigning it again can allow VSCode align field of "self.voxel_map" with SparseVoxelMapEQA
        self.voxel_map: SparseVoxelMapEQA | SparseVoxelMapProxy = voxel_map

        self.alignment_heuristics_type = alignment_heuristics_type
        self.create_collision_masks(orientation_resolution)
        self.traj = None
        self.eqa_client = GeminiClient(EQA_PROMPT, model="gemini-2.5-pro-preview-03-25")

    def query_answer(self, question: str, xyt, planner):
        """
        Util function to prompt mLLM to provide answer output, and process the raw answer output into robot's next step.
        """

        # Extract keywords from the question
        self.voxel_map.extract_relevant_objects(question)

        # messages = [{"type": "text", "text": "Question: " + question}]
        commands: List[Any] = ["Question: " + question]
        # messages.append({"type": "text", "text": "HISTORY: "})
        commands.append("HISTORY: ")
        for (i, history_output) in enumerate(self.voxel_map.history_outputs):
            # messages.append({"type": "text", "text": "Iteration_" + str(i) + ":" + history_output})
            commands.append("Iteration_" + str(i) + ":" + history_output)
        # messages.append({"role": "user", "content": [{"type": "input_text", "text": question}]})

        # Select the task relevant images with DynaMem
        img_idx = 0
        all_obs_ids = set()

        for relevant_object in self.voxel_map.relevant_objects:
            # Limit the total number of images to 6
            image_ids, _, _ = self.voxel_map.find_all_images(
                relevant_object,
                min_similarity_threshold=0.12,
                max_img_num=6 // len(self.voxel_map.relevant_objects),
                min_point_num=40,
            )
            for obs_id in image_ids:
                obs_id = int(obs_id) - 1
                all_obs_ids.add(obs_id)

        all_obs_ids = list(all_obs_ids)  # type: ignore

        # Prepare the visual clues (image descriptions)
        selected_images, action_prompt = self.get_image_descriptions(xyt, planner, all_obs_ids)
        commands.append(action_prompt)
        self.voxel_map.log_text(commands)
        relevant_images = []

        for obs_id in all_obs_ids:
            rgb = np.copy(self.voxel_map.observations[obs_id].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Log the input images
            image.save(
                self.voxel_map.log
                + "/"
                + str(len(self.voxel_map.image_descriptions))
                + "/"
                + str(img_idx)
                + ".jpg"
            )
            img_idx += 1

            commands.append(image)
            relevant_images.append(image)

        # Extract answers
        answer_outputs = (
            self.eqa_client(commands).replace("*", "").replace("/", "").replace("#", "").lower()
        )

        print(commands)
        print(answer_outputs)

        (
            reasoning,
            answer,
            confidence,
            action,
            confidence_reasoning,
        ) = self.voxel_map.parse_answer(answer_outputs)

        # If the robot is not confident, it should plan exploration
        if not confidence:
            action = selected_images[int(action) - 1]
            rgb = np.copy(self.voxel_map.observations[action - 1].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Cache conversations between the robot and the mLLM for the next iteration of question answering planning
            self.voxel_map.history_outputs.append(
                "Answer:"
                + answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction:"
                + "Navigate to Image with objects "
                + str(self.voxel_map.image_descriptions[action - 1][0])
                + " with grid coord "
                + str(self.voxel_map.image_descriptions[action - 1][1])
                + "\nConfidence reasoning:"
                + confidence_reasoning
            )
        else:
            action = None

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            self.get_target_point_from_image_id(action, xyt, planner)
            if action is not None
            else None,
            relevant_images,
        )

    def get_image_descriptions(self, xyt, planner, obs_ids):
        """
        Select visual clues of all active images (images still associated with some voxel points in the voxel map)
        """
        (
            _,
            selected_images,
            image_descriptions,
        ) = self.voxel_map.get_active_image_descriptions()
        frontier_ids = list(self.get_frontier_ids(xyt, planner))
        options = ""
        if len(image_descriptions) > 0:
            for i, (cluster, grid_coord) in enumerate(image_descriptions):
                index = selected_images[i]
                cluster_string = ""
                for ob in cluster:
                    cluster_string += ob + ", "
                cluster_string = cluster_string[:-2] + ";"
                # Indicate the grid coord this image describes to avoid redundant exploration.
                cluster_string += " This image is taken at grid coords " + str(grid_coord)
                # If we have already send the raw image observation to LLM.
                if index in obs_ids:
                    cluster_string += (
                        " This observation description is associated with Image "
                        + str(obs_ids.index(index) + 1)
                        + ";"
                    )
                # If this image corresponds to an unexplored frontier
                if index in frontier_ids:
                    cluster_string += (
                        " This observation description corresponds to unexplored space;"
                    )
                options += f"{i+1}. {cluster_string}\n"
        return selected_images, "IMAGE_DESCRIPTIONS: " + options

    def get_target_point_from_image_id(self, image_id: int, xyt, planner):
        """
        When the robot is not confident with the answer, mLLM will output an image id indicating a rough direction for the robot to take the next step.
        This function selects the target point's xy coordinate based on the image id provided.
        """

        # history output by get_active_descriptions output a history id map considering history id of the floor point
        # history_soft output by get_2d_map output a history id map excluding history id of the floor point
        # Therefore, history is generally used to select active image observations while history_soft is generally used to determine unexplored frontier
        (
            history,
            _,
            _,
        ) = self.voxel_map.get_active_image_descriptions()
        obstacles, explored = self.voxel_map.get_2d_map()
        outside_frontier = self.get_outside_frontier(xyt, planner)
        unexplored_frontier = outside_frontier & ~explored
        # Navigation priority: unexplored frontier > obstalces > others
        if torch.sum((history == image_id) & unexplored_frontier) > 0:
            print("unexplored frontier")
            image_coord = (
                ((history == image_id) & unexplored_frontier)
                .nonzero(as_tuple=False)
                .median(dim=0)
                .values.int()
            )
        elif torch.sum((history == image_id) & obstacles) > 0:
            print("obstacles")
            image_coord = (
                ((history == image_id) & obstacles)
                .nonzero(as_tuple=False)
                .median(dim=0)
                .values.int()
            )
        else:
            print("others")
            image_coord = (history == image_id).nonzero(as_tuple=False).median(dim=0).values.int()
        xy = self.voxel_map.grid_coords_to_xy(image_coord)
        return torch.Tensor([xy[0], xy[1], 1])

    def get_frontier_ids(self, xyt, planner):
        """
        This function figures out which of images correspond to an unexplored frontier.
        """
        (
            history,
            _,
            _,
        ) = self.voxel_map.get_active_image_descriptions()
        outside_frontier = self.get_outside_frontier(xyt, planner)
        _, explored = self.voxel_map.get_2d_map()
        unexplored_frontier = outside_frontier & ~explored
        history = np.ma.masked_array(history, ~unexplored_frontier)
        return np.unique(history)
