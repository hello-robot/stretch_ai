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
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from stretch.motion import Footprint

from .voxel import SparseVoxelMapProxy
from .voxel_dynamem import SparseVoxelMap
from .voxel_map import SparseVoxelMapNavigationSpace as SparseVoxelMapNavigationSpaceBase


class SparseVoxelMapNavigationSpace(SparseVoxelMapNavigationSpaceBase):

    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        voxel_map: SparseVoxelMap | SparseVoxelMapProxy,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
    ):
        super().__init__(
            voxel_map=voxel_map,
            robot=None,
            step_size=step_size,
            rotation_step_size=rotation_step_size,
            use_orientation=use_orientation,
            orientation_resolution=orientation_resolution,
            dilate_frontier_size=dilate_frontier_size,
            dilate_obstacle_size=dilate_obstacle_size,
            extend_mode=extend_mode,
        )
        self.create_collision_masks(orientation_resolution)
        self.traj = None

    def create_collision_masks(self, orientation_resolution: int):
        """Create a set of orientation masks

        Args:
            orientation_resolution: number of bins to break it into
        """
        self._footprint = Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)
        self._orientation_resolution = 64
        self._oriented_masks = []

        for i in range(orientation_resolution):
            theta = i * 2 * np.pi / orientation_resolution
            mask = self._footprint.get_rotated_mask(
                self.voxel_map.grid_resolution, angle_radians=theta
            )
            self._oriented_masks.append(mask)

    def compute_theta(self, cur_x, cur_y, end_x, end_y):
        theta = 0
        if end_x == cur_x and end_y >= cur_y:
            theta = np.pi / 2
        elif end_x == cur_x and end_y < cur_y:
            theta = -np.pi / 2
        else:
            theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
            if end_x < cur_x:
                theta = theta + np.pi
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = theta + 2 * np.pi
        return theta

    def sample_target_point(
        self, start: torch.Tensor, point: torch.Tensor, planner, exploration: bool = False
    ) -> Optional[np.ndarray]:
        """Sample a position near the mask and return.

        Args:
            look_at_any_point(bool): robot should look at the closest point on target mask instead of average pt
        """

        obstacles, explored = self.voxel_map.get_2d_map()

        # Extract edges from our explored mask
        start_pt = planner.to_pt(start)
        reachable_points = planner.get_reachable_points(start_pt)
        if len(reachable_points) == 0:
            print("No target point find, maybe no point is reachable")
            return None
        reachable_xs, reachable_ys = zip(*reachable_points)
        # # type: ignore comments used to bypass mypy check
        reachable_xs = torch.tensor(reachable_xs)  # type: ignore
        reachable_ys = torch.tensor(reachable_ys)  # type: ignore
        reachable = torch.empty(obstacles.shape, dtype=torch.bool).fill_(False)
        reachable[reachable_xs, reachable_ys] = True

        obstacles, explored = self.voxel_map.get_2d_map()
        reachable = reachable & ~obstacles

        target_x, target_y = planner.to_pt(point)

        xs, ys = torch.where(reachable)
        if len(xs) < 1:
            print("No target point find, maybe no point is reachable")
            return None
        selected_targets = torch.stack([xs, ys], dim=-1)[
            torch.linalg.norm(
                (torch.stack([xs, ys], dim=-1) - torch.tensor([target_x, target_y])).float(), dim=-1
            )
            .topk(k=len(xs), largest=False)
            .indices
        ]

        for selected_target in selected_targets:
            selected_x, selected_y = planner.to_xy([selected_target[0], selected_target[1]])
            theta = self.compute_theta(selected_x, selected_y, point[0], point[1])

            target_is_valid = self.is_valid(np.array([selected_x, selected_y, theta]))
            if not target_is_valid:
                continue
            if np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.35:
                continue
            elif np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.5:
                i = (point[0] - selected_target[0]) // abs(point[0] - selected_target[0])
                j = (point[1] - selected_target[1]) // abs(point[1] - selected_target[1])
                index_i = int(selected_target[0].int() + i)
                index_j = int(selected_target[1].int() + j)
                if obstacles[index_i][index_j]:
                    target_is_valid = False

            if not target_is_valid:
                continue

            return np.array([selected_x, selected_y, theta])

        return None

    def sample_exploration(self, xyt, planner, text=None, debug=False):
        """
        Sample an exploration target
        """
        obstacles, explored, history_soft = self.voxel_map.get_2d_map(
            return_history_id=True, kernel=5
        )
        outside_frontier = self.voxel_map.get_outside_frontier(xyt, planner)

        time_heuristics = self._time_heuristic(history_soft, outside_frontier, debug=debug)

        # TODO: Find good alignment heuristic, we have found few candidates but none of them has satisfactory performance

        ######################################
        # Candidate 1: Borrow the idea from https://arxiv.org/abs/2310.10103
        # for i, (cluster, _) in enumerate(image_descriptions):
        #   cluser_string = ""
        #   for ob in cluster:
        #       cluser_string += ob + ", "
        #   options += f"{i+1}. {cluser_string[:-2]}\n"

        # if positive:
        #     messages = f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I search next if I try to {task}?"
        #     choices = self.positive_score_client.sample(messages, n_samples=num_samples)
        # else:
        #     messages = f"I observe the following clusters of objects while exploring the room:\n\n {options}\nWhere should I avoid spending time searching if I try to {task}?"
        #     choices = self.negative_score_client.sample(messages, n_samples=num_samples)

        # answers = []
        # reasonings = []
        # for choice in choices:
        #     complete_response = choice.lower()
        #     reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
        #     # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
        #     if len(complete_response.split("answer:")) > 1:
        #          answer = complete_response.split("answer:")[1].split("\n")[0]
        #          # Separate the answers by commas
        #          answers.append([int(x) for x in answer.split(",")])
        #      else:
        #          answers.append([])
        #      reasonings.append(reasoning)

        # # Flatten answers
        # flattened_answers = [item for sublist in answers for item in sublist]
        # filtered_flattened_answers = [
        #     x for x in flattened_answers if x >= 1 and x <= len(image_descriptions)
        # ]
        # # Aggregate into counts and normalize to probabilities
        # answer_counts = {
        #     x: filtered_flattened_answers.count(x) / len(answers)
        #     for x in set(filtered_flattened_answers)
        # }
        ######################################
        # Candidate 2: Naively use semantic feature alignment
        # def get_2d_alignment_heuristics(self, text: str, debug: bool = False):
        # if self.semantic_memory._points is None:
        #     return None
        # # Convert metric measurements to discrete
        # # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        # xyz, _, _, _ = self.semantic_memory.get_pointcloud()
        # xyz = xyz.detach().cpu()
        # if xyz is None:
        #     xyz = torch.zeros((0, 3))

        # device = xyz.device
        # xyz = ((xyz / self.grid_resolution) + self.grid_origin).long()
        # xyz[xyz[:, -1] < 0, -1] = 0

        # # Crop to robot height
        # min_height = int(self.obs_min_height / self.grid_resolution)
        # max_height = int(self.obs_max_height / self.grid_resolution)
        # grid_size = self.grid_size + [max_height]

        # # Mask out obstacles only above a certain height
        # obs_mask = xyz[:, -1] < max_height
        # xyz = xyz[obs_mask, :]
        # alignments = self.find_alignment_over_model(text)[0].detach().cpu()
        # alignments = alignments[obs_mask][:, None]

        # alignment_heuristics = scatter3d(xyz, alignments, grid_size, "max")
        # alignment_heuristics = torch.max(alignment_heuristics, dim=-1).values
        # alignment_heuristics = torch.from_numpy(
        #     maximum_filter(alignment_heuristics.numpy(), size=5)
        # )

        alignments_heuristics = None
        total_heuristics = time_heuristics

        rounded_heuristics = np.ceil(total_heuristics * 200) / 200
        max_heuristic = rounded_heuristics.max()
        indices = np.column_stack(np.where(rounded_heuristics == max_heuristic))
        closest_index = np.argmin(np.linalg.norm(indices - np.asarray(planner.to_pt(xyt)), axis=-1))
        index = indices[closest_index]
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

    def _time_heuristic(
        self, history_soft, outside_frontier, time_smooth=0.1, time_threshold=10, debug=False
    ):
        history_soft = np.ma.masked_array(history_soft, ~outside_frontier)
        time_heuristics = history_soft.max() - history_soft
        time_heuristics[history_soft < 1] = float("inf")
        time_heuristics = 1 / (1 + np.exp(-time_smooth * (time_heuristics - time_threshold)))
        index = np.unravel_index(np.argmax(time_heuristics), history_soft.shape)
        # return index
        # debug = True
        if debug:
            # plt.clf()
            plt.title("time")
            plt.imshow(history_soft)
            plt.scatter(index[1], index[0], s=15, c="r")
            plt.show()
        return time_heuristics

    def to_pt(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts a point from continuous, world xy coordinates to grid coordinates.

        Args:
            xy: The point in continuous xy coordinates.

        Returns:
            The point in discrete grid coordinates.
        """
        # # type: ignore to bypass mypy checking
        xy = np.array([xy[0], xy[1]])  # type: ignore
        pt = self.voxel_map.xy_to_grid_coords(xy)  # type: ignore
        return int(pt[0]), int(pt[1])

    def to_xy(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        """Converts a point from grid coordinates to continuous, world xy coordinates.

        Args:
            pt: The point in grid coordinates.

        Returns:
            The point in continuous xy coordinates.
        """
        # # type: ignore to bypass mypy checking
        pt = np.array([pt[0], pt[1]])  # type: ignore
        xy = self.voxel_map.grid_coords_to_xy(pt)  # type: ignore
        return float(xy[0]), float(xy[1])

    def sample_navigation(self, start, planner, point, mode="navigation"):
        plt.clf()
        if point is None:
            start_pt = self.to_pt(start)
            return None
        goal = self.sample_target_point(start, point, planner, exploration=mode != "navigation")
        print("point:", point, "goal:", goal)
        obstacles, explored = self.voxel_map.get_2d_map()
        plt.imshow(obstacles)
        start_pt = self.to_pt(start)
        plt.scatter(start_pt[1], start_pt[0], s=15, c="b")
        point_pt = self.to_pt(point)
        plt.scatter(point_pt[1], point_pt[0], s=15, c="r")
        if goal is not None:
            goal_pt = self.to_pt(goal)
            plt.scatter(goal_pt[1], goal_pt[0], s=10, c="g")
        # plt.show()
        return goal

    def sample_frontier(self, planner, start_pose=[0, 0, 0], text=None):
        (
            index,
            time_heuristics,
            alignments_heuristics,
            total_heuristics,
        ) = self.sample_exploration(
            start_pose,
            planner,
            text=text,
            debug=False,
        )

        obstacles, explored = self.voxel_map.get_2d_map()
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))
