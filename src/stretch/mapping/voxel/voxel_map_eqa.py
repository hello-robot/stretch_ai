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
from stretch.utils.morphology import get_edges

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
        self.voxel_map.extract_relevant_objects(question)

        # messages = [{"type": "text", "text": "Question: " + question}]
        commands: List[Any] = ["Question: " + question]
        # messages.append({"type": "text", "text": "HISTORY: "})
        commands.append("HISTORY: ")
        for (i, history_output) in enumerate(self.voxel_map.history_outputs):
            # messages.append({"type": "text", "text": "Iteration_" + str(i) + ":" + history_output})
            commands.append("Iteration_" + str(i) + ":" + history_output)
        # messages.append({"role": "user", "content": [{"type": "input_text", "text": question}]})

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

        selected_images, action_prompt = self.get_image_descriptions(xyt, planner, all_obs_ids)
        commands.append(action_prompt)
        self.voxel_map.log_text(commands)
        relevant_images = []

        for obs_id in all_obs_ids:
            rgb = np.copy(self.voxel_map.observations[obs_id].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

            # Save the input images
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

        if not confidence:
            action = selected_images[int(action) - 1]
            rgb = np.copy(self.voxel_map.observations[action - 1].rgb.numpy())
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            # image.show()

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

        # Debug answer and confidence
        # print("Answer:", answer)
        # print("Confidence:", confidence)
        # print("Answer outputs:", answer_outputs)

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
        (
            history,
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
        # history outpyt by get_active_descriptions output a history id map considering history id of the floor point
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
        # from matplotlib import pyplot as plt
        # plt.clf()
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

    def get_outside_frontier(self, xyt, planner):
        obstacles, _ = self.voxel_map.get_2d_map()
        if len(xyt) == 3:
            xyt = xyt[:2]
        reachable_points = planner.get_reachable_points(planner.to_pt(xyt))
        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = torch.tensor(reachable_xs)
        reachable_ys = torch.tensor(reachable_ys)

        reachable_map = torch.zeros_like(obstacles)
        reachable_map[reachable_xs, reachable_ys] = 1
        reachable_map = reachable_map.to(torch.bool)
        edges = get_edges(reachable_map)
        expanded_frontier = edges
        return expanded_frontier & ~reachable_map

    def sample_exploration(
        self, xyt, planner, use_alignment_heuristics=True, text=None, debug=False, verbose=True
    ):
        obstacles, explored, history_soft = self.voxel_map.get_2d_map(
            return_history_id=True, kernel=5
        )
        outside_frontier = self.get_outside_frontier(xyt, planner)

        time_heuristics = self._time_heuristic(history_soft, outside_frontier, debug=debug)
        if (
            use_alignment_heuristics
            and len(self.voxel_map.semantic_memory._points) > 0
            and text != ""
            and text is not None
        ):
            if self.alignment_heuristics_type == "encoder":
                alignments_heuristics = self.voxel_map.get_2d_alignment_heuristics(text)
                alignments_heuristics = self._alignment_heuristic(
                    alignments_heuristics, outside_frontier, debug=debug
                )
                total_heuristics = time_heuristics + 0.3 * alignments_heuristics
            elif self.alignment_heuristics_type == "mllm":
                alignments_heuristics = self.voxel_map.get_2d_alignment_heuristics_mllm(text)
                alignments_heuristics = np.ma.masked_array(alignments_heuristics, ~outside_frontier)
                total_heuristics = time_heuristics + alignments_heuristics
                if verbose:
                    import matplotlib
                    from matplotlib import pyplot as plt

                    matplotlib.use("Agg")
                    plt.close("all")
                    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                    axs[0, 0].imshow(obstacles)
                    axs[0, 0].set_title("obstacle map")
                    axs[0, 1].imshow(alignments_heuristics)
                    axs[0, 1].set_title("exploration alignment heuristics")
                    axs[1, 0].imshow(time_heuristics)
                    axs[1, 0].set_title("time heuristics")
                    axs[1, 1].imshow(total_heuristics)
                    axs[1, 1].set_title("total heuristics")
                    for ax in axs.flat:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        self.voxel_map.log
                        + "/exploration"
                        + str(len(self.voxel_map.image_descriptions))
                        + ".jpg",
                        dpi=300,
                    )
                    # plt.close("all")
                    # plt.imshow(obstacles)
                    # plt.show()
                    # plt.imshow(alignments_heuristics)
                    # plt.show()
                    # plt.imshow(time_heuristics)
                    # plt.show()
                    # plt.imshow(total_heuristics)
                    # plt.show()
            else:
                raise ValueError(
                    f"Invalid alignment heuristics type: {self.alignment_heuristics_type}"
                )
        else:
            alignments_heuristics = None
            total_heuristics = time_heuristics

        rounded_heuristics = np.ceil(total_heuristics * 200) / 200
        max_heuristic = rounded_heuristics.max()
        indices = np.column_stack(np.where(rounded_heuristics == max_heuristic))
        closest_index = np.argmin(np.linalg.norm(indices - np.asarray(planner.to_pt(xyt)), axis=-1))
        index = indices[closest_index]
        # index = np.unravel_index(np.argmax(total_heuristics), total_heuristics.shape)
        # debug = True
        if debug:
            from matplotlib import pyplot as plt

            plt.subplot(221)
            plt.imshow(obstacles.int() * 5 + outside_frontier.int() * 10)
            plt.subplot(222)
            plt.imshow(explored.int() * 5)
            plt.subplot(223)
            plt.imshow(total_heuristics)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.subplot(224)
            plt.imshow(history_soft)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.show()
        return index, time_heuristics, alignments_heuristics, total_heuristics
