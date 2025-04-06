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

import numpy as np
import torch

from stretch.utils.morphology import binary_dilation, get_edges

from .voxel import SparseVoxelMapProxy
from .voxel_eqa import SparseVoxelMap
from .voxel_map_dynamem import SparseVoxelMapNavigationSpace as SparseVoxelMapNavigationSpaceBase


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
        self.alignment_heuristics_type = alignment_heuristics_type
        self.create_collision_masks(orientation_resolution)
        self.traj = None

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
                # from matplotlib import pyplot as plt
                # plt.clf()
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
