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
import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skfmm
import skimage
import skimage.morphology
import torch

from stretch.dynav.mapping_utils.voxel import SparseVoxelMap
from stretch.motion import XYT, Footprint
from stretch.utils.geometry import angle_difference, interpolate_angles
from stretch.utils.morphology import (
    binary_dilation,
    binary_erosion,
    expand_mask,
    find_closest_point_on_mask,
    get_edges,
)


class SparseVoxelMapNavigationSpace(XYT):
    """subclass for sampling XYT states from explored space"""

    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        voxel_map: SparseVoxelMap,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
    ):
        self.step_size = step_size
        self.rotation_step_size = rotation_step_size
        self.voxel_map = voxel_map
        self.create_collision_masks(orientation_resolution)
        self.extend_mode = extend_mode

        # Always use 3d states
        self.use_orientation = use_orientation
        if self.use_orientation:
            self.dof = 3
        else:
            self.dof = 2

        # # type: ignore comments used to bypass mypy check
        self._kernels = {}  # type: ignore

        if dilate_frontier_size > 0:
            self.dilate_explored_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(dilate_frontier_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_explored_kernel = None
        if dilate_obstacle_size > 0:
            self.dilate_obstacles_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(dilate_obstacle_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_obstacles_kernel = None

    def draw_state_on_grid(
        self, img: np.ndarray, state: np.ndarray, weight: int = 10
    ) -> np.ndarray:
        """Helper function to draw masks on image"""
        grid_xy = self.voxel_map.xy_to_grid_coords(state[:2])
        mask = self.get_oriented_mask(state[2])
        x0 = int(np.round(float(grid_xy[0] - mask.shape[0] // 2)))
        x1 = x0 + mask.shape[0]
        y0 = int(np.round(float(grid_xy[1] - mask.shape[1] // 2)))
        y1 = y0 + mask.shape[1]
        img[x0:x1, y0:y1] += mask * weight
        return img

    def create_collision_masks(self, orientation_resolution: int, show_all: bool = False):
        """Create a set of orientation masks

        Args:
            orientation_resolution: number of bins to break it into
        """
        self._footprint = Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1)
        self._orientation_resolution = 64
        self._oriented_masks = []

        # NOTE: this is just debug code - lets you see what the masks look like
        assert not show_all or orientation_resolution == 64

        for i in range(orientation_resolution):
            theta = i * 2 * np.pi / orientation_resolution
            mask = self._footprint.get_rotated_mask(
                self.voxel_map.grid_resolution, angle_radians=theta
            )
            if show_all:
                plt.subplot(8, 8, i + 1)
                plt.axis("off")
                plt.imshow(mask.cpu().numpy())
            self._oriented_masks.append(mask)
        if show_all:
            plt.show()

    def distance(self, q0: np.ndarray, q1: np.ndarray) -> float:
        """Return distance between q0 and q1."""
        assert len(q0) == 3, "must use 3 dimensions for current state"
        assert len(q1) == 3 or len(q1) == 2, "2 or 3 dimensions for goal"
        if len(q1) == 3:
            # Measure to the final position exactly
            return np.linalg.norm(q0 - q1).item()
        else:
            # Measure only to the final goal x/y position
            return np.linalg.norm(q0[:2] - q1[:2]).item()

    def extend(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """extend towards another configuration in this space. Will be either separate or joint depending on if the robot can "strafe":
        separate: move then rotate
        joint: move and rotate all at once."""
        assert len(q0) == 3, f"initial configuration must be 3d, was {q0}"
        assert len(q1) == 3 or len(q1) == 2, f"final configuration can be 2d or 3d, was {q1}"
        if self.extend_mode == "separate":
            return self._extend_separate(q0, q1)
        elif self.extend_mode == "joint":
            # Just default to linear interpolation, does not use rotation_step_size
            return super().extend(q0, q1)
        else:
            raise NotImplementedError(f"not supported: {self.extend_mode=}")

    def _extend_separate(self, q0: np.ndarray, q1: np.ndarray, xy_tol: float = 1e-8):
        """extend towards another configuration in this space.
        TODO: we can set the classes here, right now assuming still np.ndarray"""
        assert len(q0) == 3, f"initial configuration must be 3d, was {q0}"
        assert len(q1) == 3 or len(q1) == 2, f"final configuration can be 2d or 3d, was {q1}"
        dxy = q1[:2] - q0[:2]
        step = dxy / np.linalg.norm(dxy + self.tolerance) * self.step_size
        xy = np.copy(q0[:2])
        goal_dxy = np.linalg.norm(q1[:2] - q0[:2])
        if (
            goal_dxy
            > xy_tol
            # or goal_dxy > self.step_size
            # or angle_difference(q1[-1], q0[-1]) > self.rotation_step_size
        ):
            # Turn to new goal
            # Compute theta looking at new goal point
            new_theta = math.atan2(dxy[1], dxy[0])
            if new_theta < 0:
                new_theta += 2 * np.pi

            # TODO: orient towards the new theta
            cur_theta = q0[-1]
            angle_diff = angle_difference(new_theta, cur_theta)
            while angle_diff > self.rotation_step_size:
                # Interpolate
                cur_theta = interpolate_angles(cur_theta, new_theta, self.rotation_step_size)
                # print("interp ang =", cur_theta, "from =", cur_theta, "to =", new_theta)
                yield np.array([xy[0], xy[1], cur_theta])
                angle_diff = angle_difference(new_theta, cur_theta)

            # First, turn in the right direction
            next_pt = np.array([xy[0], xy[1], new_theta])
            # After this we should have finished turning
            yield next_pt

            # Now take steps towards the right goal
            while np.linalg.norm(xy - q1[:2]) > self.step_size:
                xy = xy + step
                yield np.array([xy[0], xy[1], new_theta])

            # Update current angle
            cur_theta = new_theta

            # Finish stepping to goal
            xy[:2] = q1[:2]
            yield np.array([xy[0], xy[1], cur_theta])
        else:
            cur_theta = q0[-1]

        # now interpolate to goal angle
        angle_diff = angle_difference(q1[-1], cur_theta)
        while angle_diff > self.rotation_step_size:
            # Interpolate
            cur_theta = interpolate_angles(cur_theta, q1[-1], self.rotation_step_size)
            yield np.array([xy[0], xy[1], cur_theta])
            angle_diff = angle_difference(q1[-1], cur_theta)

        # Get to final angle
        yield np.array([xy[0], xy[1], q1[-1]])

        # At the end, rotate into the correct orientation
        yield q1

    def _get_theta_index(self, theta: float) -> int:
        """gets the index associated with theta here"""
        if theta < 0:
            theta += 2 * np.pi
        if theta >= 2 * np.pi:
            theta -= 2 * np.pi
        assert theta >= 0 and theta <= 2 * np.pi, "only angles between 0 and 2*PI allowed"
        theta_idx = np.round((theta / (2 * np.pi) * self._orientation_resolution) - 0.5)
        if theta_idx == self._orientation_resolution:
            theta_idx = 0
        return int(theta_idx)

    def get_oriented_mask(self, theta: float) -> torch.Tensor:
        theta_idx = self._get_theta_index(theta)
        return self._oriented_masks[theta_idx]

    def is_valid(
        self,
        state: Union[np.ndarray, torch.Tensor, List],
        is_safe_threshold=1.0,
        debug: bool = False,
        verbose: bool = False,
    ) -> bool:
        """Check to see if state is valid; i.e. if there's any collisions if mask is at right place"""
        assert len(state) == 3
        if isinstance(state, torch.Tensor):
            state = state.float().numpy()
        state = np.array(state)
        ok = bool(self.voxel_map.xyt_is_safe(state[:2]))
        # if verbose:
        #     print('is navigable:', ok)
        if not ok:
            # This was
            return False

        # Now sample mask at this location
        mask = self.get_oriented_mask(state[-1])
        assert mask.shape[0] == mask.shape[1], "square masks only for now"
        dim = mask.shape[0]
        half_dim = dim // 2
        grid_xy = self.voxel_map.xy_to_grid_coords(state[:2])
        x0 = int(grid_xy[0]) - half_dim
        x1 = x0 + dim
        y0 = int(grid_xy[1]) - half_dim
        y1 = y0 + dim

        obstacles, explored = self.voxel_map.get_2d_map()

        crop_obs = obstacles[x0:x1, y0:y1]
        crop_exp = explored[x0:x1, y0:y1]
        assert mask.shape == crop_obs.shape
        assert mask.shape == crop_exp.shape

        collision = torch.any(crop_obs & mask)

        p_is_safe = (torch.sum((crop_exp & mask) | ~mask) / (mask.shape[0] * mask.shape[1])).item()
        is_safe = p_is_safe >= is_safe_threshold
        if verbose:
            print(f"{collision=}, {is_safe=}, {p_is_safe=}, {is_safe_threshold=}")

        valid = bool((not collision) and is_safe)
        if debug:
            if collision:
                print("- state in collision")
            if not is_safe:
                print("- not safe")

            print(f"{valid=}")
            obs = obstacles.cpu().numpy().copy()
            exp = explored.cpu().numpy().copy()
            obs[x0:x1, y0:y1] = 1
            plt.subplot(321)
            plt.imshow(obs)
            plt.subplot(322)
            plt.imshow(exp)
            plt.subplot(323)
            plt.imshow(crop_obs.cpu().numpy())
            plt.title("obstacles")
            plt.subplot(324)
            plt.imshow(crop_exp.cpu().numpy())
            plt.title("explored")
            plt.subplot(325)
            plt.imshow(mask.cpu().numpy())
            plt.show()

        return valid

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
        # if self.dilate_obstacles_kernel is not None:
        #     obstacles = binary_dilation(
        #         obstacles.float().unsqueeze(0).unsqueeze(0), self.dilate_obstacles_kernel
        #     )[0, 0].bool()
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

        # TODO: was this:
        # expanded_mask = expanded_mask & less_explored & ~obstacles

        for selected_target in selected_targets:
            selected_x, selected_y = planner.to_xy([selected_target[0], selected_target[1]])
            theta = self.compute_theta(selected_x, selected_y, point[0], point[1])

            # if debug and self.is_valid([selected_x, selected_y, theta]):
            #     import matplotlib.pyplot as plt

            #     obstacles, explored = self.voxel_map.get_2d_map()
            #     plt.scatter(ys, xs, s = 1)
            #     plt.scatter(selected_target[1], selected_target[0], s = 10)
            #     plt.scatter(target_y, target_x, s = 10)
            #     plt.imshow(obstacles)
            target_is_valid = self.is_valid([selected_x, selected_y, theta])
            # print('Target:', [selected_x, selected_y, theta])
            # print('Target is valid:', target_is_valid)
            if not target_is_valid:
                continue
            if np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.35:
                continue
            elif np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.45:
                # print('OBSTACLE AVOIDANCE')
                # print(selected_target[0].int(), selected_target[1].int())
                i = (point[0] - selected_target[0]) // abs(point[0] - selected_target[0])
                j = (point[1] - selected_target[1]) // abs(point[1] - selected_target[1])
                index_i = int(selected_target[0].int() + i)
                index_j = int(selected_target[1].int() + j)
                if obstacles[index_i][index_j]:
                    target_is_valid = False
            # elif np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= 0.5:
            #     for i in [-1, 0, 1]:
            #         for j in [-1, 0, 1]:
            #             if obstacles[selected_target[0] + i][selected_target[1] + j]:
            #                 target_is_valid = False
            if not target_is_valid:
                continue

            return np.array([selected_x, selected_y, theta])
        return None

    def sample_near_mask(
        self,
        mask: torch.Tensor,
        radius_m: float = 0.6,
        max_tries: int = 1000,
        verbose: bool = False,
        debug: bool = False,
        look_at_any_point: bool = False,
    ):
        """Sample a position near the mask and return.

        Args:
            look_at_any_point(bool): robot should look at the closest point on target mask instead of average pt
        """

        obstacles, explored = self.voxel_map.get_2d_map()

        # Extract edges from our explored mask

        # Radius computed from voxel map measurements
        radius = np.ceil(radius_m / self.voxel_map.grid_resolution)
        expanded_mask = expand_mask(mask, radius)

        # TODO: was this:
        # expanded_mask = expanded_mask & less_explored & ~obstacles
        expanded_mask = expanded_mask & explored & ~obstacles
        # print(torch.where(explored & ~obstacles))
        # print(torch.where(expanded_mask))

        if debug:
            import matplotlib.pyplot as plt

            plt.imshow(
                mask.int() * 20 + expanded_mask.int() * 10 + explored.int() + obstacles.int() * 5
            )
            # import datetime
            # current_datetime = datetime.datetime.now()
            # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            # plt.savefig('debug_' + formatted_datetime + '.png')

        # Where can the robot go?
        valid_indices = torch.nonzero(expanded_mask, as_tuple=False)
        if valid_indices.size(0) == 0:
            print("[VOXEL MAP: sampling] No valid goals near mask!")
            return None
        if not look_at_any_point:
            mask_indices = torch.nonzero(mask, as_tuple=False)
            outside_point = mask_indices.float().mean(dim=0)

        # maximum number of tries
        for i in range(max_tries):
            random_index = torch.randint(valid_indices.size(0), (1,))
            point_grid_coords = valid_indices[random_index]

            if look_at_any_point:
                outside_point = find_closest_point_on_mask(mask, point_grid_coords.float())

            # convert back
            point = self.voxel_map.grid_coords_to_xy(point_grid_coords.numpy())
            if point is None:
                print("[VOXEL MAP: sampling] ERR:", point, point_grid_coords)
                continue
            if outside_point is None:
                print(
                    "[VOXEL MAP: sampling] ERR finding closest pt:",
                    point,
                    point_grid_coords,
                    "closest =",
                    outside_point,
                )
                continue
            theta = math.atan2(
                outside_point[1] - point_grid_coords[0, 1],
                outside_point[0] - point_grid_coords[0, 0],
            )

            # Ensure angle is in 0 to 2 * PI
            if theta < 0:
                theta += 2 * np.pi

            xyt = torch.zeros(3)
            # # type: ignore to bypass mypy check
            xyt[:2] = point  # type: ignore
            xyt[2] = theta

            # Check to see if this point is valid
            if verbose:
                print("[VOXEL MAP: sampling]", radius, i, "sampled", xyt)
            if self.is_valid(xyt, verbose=verbose):
                yield xyt

        # We failed to find anything useful
        return None

    def has_zero_contour(self, phi):
        """
        Check if a zero contour exists in the given phi array.

        Parameters:
        - phi: 2D NumPy array with boolean values.

        Returns:
        - True if a zero contour exists, False otherwise.
        """
        # Check if there are True and False values in the array
        has_true_values = np.any(phi)
        has_false_values = np.any(~phi)

        # Return True if both True and False values are present
        return has_true_values and has_false_values

    def _get_kernel(self, size: int):
        """Return a kernel for expanding/shrinking areas."""
        if size <= 0:
            return None
        if size not in self._kernels:
            kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(size)).unsqueeze(0).unsqueeze(0).float(),
                requires_grad=False,
            )
            self._kernels[size] = kernel
        return self._kernels[size]

    def get_frontier(
        self, expand_size: int = 5, debug: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute frontier regions of the map"""

        obstacles, explored = self.voxel_map.get_2d_map()
        # Extract edges from our explored mask
        # if self.dilate_obstacles_kernel is not None:
        #     obstacles = binary_dilation(
        #         obstacles.float().unsqueeze(0).unsqueeze(0), self.dilate_obstacles_kernel
        #     )[0, 0].bool()
        if self.dilate_explored_kernel is not None:
            less_explored = binary_erosion(
                explored.float().unsqueeze(0).unsqueeze(0), self.dilate_explored_kernel
            )[0, 0]
        else:
            less_explored = explored

        # Get the masks from our 3d map
        edges = get_edges(less_explored)

        # Do not explore obstacles any more
        traversible = explored & ~obstacles
        frontier_edges = edges & ~obstacles

        kernel = self._get_kernel(expand_size)
        if kernel is not None:
            expanded_frontier = binary_dilation(
                frontier_edges.float().unsqueeze(0).unsqueeze(0),
                kernel,
            )[0, 0].bool()
        else:
            # This is a bad idea, planning will probably fail
            expanded_frontier = frontier_edges

        outside_frontier = expanded_frontier & ~explored
        frontier = expanded_frontier & ~obstacles & explored

        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(321)
            plt.imshow(obstacles.cpu().numpy())
            plt.title("obstacles")
            plt.subplot(322)
            plt.imshow(explored.bool().cpu().numpy())
            plt.title("explored")
            plt.subplot(323)
            plt.imshow((traversible & frontier).cpu().numpy())
            plt.title("traversible & frontier")
            plt.subplot(324)
            plt.imshow((expanded_frontier).cpu().numpy())
            plt.title("just frontiers")
            plt.subplot(325)
            plt.imshow((edges).cpu().numpy())
            plt.title("edges")
            plt.subplot(326)
            plt.imshow((frontier_edges).cpu().numpy())
            plt.title("frontier_edges")
            # plt.show()

        return frontier, outside_frontier, traversible

    def sample_exploration(
        self,
        xyt,
        planner,
        voxel_map_localizer=None,
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
        if voxel_map_localizer is not None:
            alignments_heuristics = self.voxel_map.get_2d_alignment_heuristics(
                voxel_map_localizer, text
            )
            alignments_heuristics = self._alignment_heuristic(
                alignments_heuristics, outside_frontier, debug=debug
            )
            total_heuristics = time_heuristics + alignments_heuristics
        else:
            alignments_heuristics = None
            total_heuristics = time_heuristics

        rounded_heuristics = np.ceil(total_heuristics * 100) / 100
        max_heuristic = rounded_heuristics.max()
        indices = np.column_stack(np.where(rounded_heuristics == max_heuristic))
        closest_index = np.argmin(
            np.linalg.norm(indices - np.asarray(planner.to_pt([0, 0, 0])), axis=-1)
        )
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
            plt.show()
        return index, time_heuristics, alignments_heuristics, total_heuristics

    def _alignment_heuristic(
        self,
        alignments,
        outside_frontier,
        alignment_smooth=50,
        alignment_threshold=0.12,
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
        self, history_soft, outside_frontier, time_smooth=0.1, time_threshold=15, debug=False
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

    def sample_closest_frontier(
        self,
        xyt: np.ndarray,
        max_tries: int = 1000,
        expand_size: int = 5,
        debug: bool = False,
        verbose: bool = False,
        step_dist: float = 0.1,
        min_dist: float = 0.1,
    ):
        """Sample a valid location on the current frontier using FMM planner to compute geodesic distance. Returns points in order until it finds one that's valid.

        Args:
            xyt(np.ndrray): [x, y, theta] of the agent; must be of size 2 or 3.
            max_tries(int): number of attempts to make for rejection sampling
            debug(bool): show visualizations of frontiers
            step_dist(float): how far apart in geo dist these points should be
        """
        assert len(xyt) == 2 or len(xyt) == 3, f"xyt must be of size 2 or 3 instead of {len(xyt)}"

        frontier, outside_frontier, traversible = self.get_frontier(
            expand_size=expand_size, debug=debug
        )

        # from scipy.ndimage.morphology import distance_transform_edt
        m = np.ones_like(traversible)
        start_x, start_y = self.voxel_map.xy_to_grid_coords(xyt[:2]).int().cpu().numpy()
        if verbose or debug:
            print("--- Coordinates ---")
            print(f"{xyt=}")
            print(f"{start_x=}, {start_y=}")

        m[start_x, start_y] = 0
        m = np.ma.masked_array(m, ~traversible)

        if not self.has_zero_contour(m):
            if verbose:
                print("traversible frontier had zero contour! no where to go.")
            return None

        distance_map = skfmm.distance(m, dx=1)
        frontier_map = distance_map.copy()
        # Masks are the areas we are ignoring - ignore everything but the frontiers
        frontier_map.mask = np.bitwise_or(frontier_map.mask, ~frontier.cpu().numpy())

        # Get distances of frontier points
        distances = frontier_map.compressed()
        xs, ys = np.where(~frontier_map.mask)

        if verbose or debug:
            print(f"-> found {len(distances)} items")

        assert len(xs) == len(ys) and len(xs) == len(distances)
        tries = 1
        prev_dist = -1 * float("Inf")
        for x, y, dist in sorted(zip(xs, ys, distances), key=lambda x: x[2]):
            if dist < min_dist:
                continue

            # Don't explore too close to where we are
            if dist < prev_dist + step_dist:
                continue
            prev_dist = dist

            point_grid_coords = torch.FloatTensor([[x, y]])
            outside_point = find_closest_point_on_mask(outside_frontier, point_grid_coords)

            if outside_point is None:
                print(
                    "[VOXEL MAP: sampling] ERR finding closest pt:",
                    point_grid_coords,
                    "closest =",
                    outside_point,
                )
                continue

            yield self.voxel_map.grid_coords_to_xy(outside_point)
            # # convert back to real-world coordinates
            # point = self.voxel_map.grid_coords_to_xy(point_grid_coords)
            # if point is None:
            #     print("[VOXEL MAP: sampling] ERR:", point, point_grid_coords)
            #     continue

            # theta = math.atan2(
            #     outside_point[1] - point_grid_coords[0, 1],
            #     outside_point[0] - point_grid_coords[0, 0],
            # )
            # if debug:
            #     print(f"{dist=}, {x=}, {y=}, {theta=}")

            # # Ensure angle is in 0 to 2 * PI
            # if theta < 0:
            #     theta += 2 * np.pi

            # xyt = torch.zeros(3)
            # xyt[:2] = point
            # xyt[2] = theta

            # # Check to see if this point is valid
            # if verbose:
            #     print("[VOXEL MAP: sampling] sampled", xyt)
            # if self.is_valid(xyt, debug=debug):
            #     yield xyt

            # tries += 1
            # if tries > max_tries:
            #     break
        yield None

    def show(
        self,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        backend: str = "open3d",
    ):
        """Tool for debugging map representations that we have created"""
        geoms = self.voxel_map._get_open3d_geometries(orig, norm)

        # lazily import open3d - it's a tough dependency
        import open3d

        # Show the geometries of where we have explored
        open3d.visualization.draw_geometries(geoms)
