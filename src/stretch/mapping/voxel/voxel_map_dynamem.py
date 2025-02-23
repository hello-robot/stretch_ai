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
from stretch.utils.morphology import binary_dilation, get_edges

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

    def sample_exploration(
        self,
        xyt,
        planner,
        use_alignment_heuristics=True,
        text=None,
        debug=False,
    ):
        obstacles, explored, history_soft = self.voxel_map.get_2d_map(return_history_id=True)
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
        # kernel = self._get_kernel(expand_size)
        kernel = None
        if kernel is not None:
            expanded_frontier = binary_dilation(
                edges.float().unsqueeze(0).unsqueeze(0),
                kernel,
            )[0, 0].bool()
        else:
            expanded_frontier = edges
        outside_frontier = expanded_frontier & ~reachable_map
        time_heuristics = self._time_heuristic(history_soft, outside_frontier, debug=debug)
        if (
            use_alignment_heuristics
            and len(self.voxel_map.semantic_memory._points) > 0
            and text != ""
            and text is not None
        ):
            alignments_heuristics = self.voxel_map.get_2d_alignment_heuristics(text)
            alignments_heuristics = self._alignment_heuristic(
                alignments_heuristics, outside_frontier, debug=debug
            )
            total_heuristics = time_heuristics + 0.3 * alignments_heuristics
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

    def _alignment_heuristic(
        self,
        alignments,
        outside_frontier,
        alignment_smooth=15,
        alignment_threshold=0.13,
        debug=False,
    ):
        alignments = np.ma.masked_array(alignments, ~outside_frontier)
        alignment_heuristics = 1 / (
            1 + np.exp(-alignment_smooth * (alignments - alignment_threshold))
        )
        index = np.unravel_index(np.argmax(alignment_heuristics), alignments.shape)
        if debug:
            plt.clf()
            plt.title("alignment")
            plt.imshow(alignment_heuristics)
            plt.scatter(index[1], index[0], s=15, c="g")
            plt.show()
        return alignment_heuristics

    def _time_heuristic(
        self, history_soft, outside_frontier, time_smooth=0.1, time_threshold=50, debug=False
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
            plt.imshow(time_heuristics)
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
        if text is not None and text != "":
            (
                index,
                time_heuristics,
                alignments_heuristics,
                total_heuristics,
            ) = self.sample_exploration(
                start_pose,
                planner,
                use_alignment_heuristics=True,
                text=text,
                debug=False,
            )
        else:
            index, time_heuristics, _, total_heuristics = self.sample_exploration(
                start_pose,
                planner,
                use_alignment_heuristics=False,
                text=None,
                debug=False,
            )
            alignments_heuristics = time_heuristics

        obstacles, explored = self.voxel_map.get_2d_map()
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))
