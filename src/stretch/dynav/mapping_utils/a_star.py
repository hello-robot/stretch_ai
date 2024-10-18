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

import heapq
import math
import time
from typing import List, Set, Tuple

import numpy as np

from stretch.dynav.mapping_utils.voxel_map import SparseVoxelMapNavigationSpace
from stretch.motion import PlanResult


class Node:
    """Stores an individual spot in the tree"""

    def __init__(self, state):
        self.state = state


def neighbors(pt: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [
        (pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)
    ]


class AStar:
    """Define RRT planning problem and parameters"""

    def __init__(
        self,
        space: SparseVoxelMapNavigationSpace,
    ):
        """Create RRT planner with configuration"""
        self.space = space
        self.reset()

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
            # move theta into [-pi, pi] range, for this function specifically,
            # (theta -= 2 * pi) or (theta += 2 * pi) is enough
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < np.pi:
                theta = theta + 2 * np.pi
        return theta

    def reset(self):
        # print('loading the up to date navigable map')
        # print('Wait')
        obs, exp = self.space.voxel_map.get_2d_map()
        # print('up to date navigable map loaded')
        self._navigable = ~obs & exp
        # self._navigable = ~obs
        self.start_time = time.time()

    def point_is_occupied(self, x: int, y: int) -> bool:
        """Checks if a point is occupied.

        Args:
            x: The x coordinate.
            y: The y coordinate.

        Returns:
            Whether the point is occupied.
        """
        return not bool(self._navigable[x][y])

    def to_pt(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts a point from continuous, world xy coordinates to grid coordinates.

        Args:
            xy: The point in continuous xy coordinates.

        Returns:
            The point in discrete grid coordinates.
        """
        # # type: ignore to bypass mypy checking
        xy = np.array([xy[0], xy[1]])  # type: ignore
        pt = self.space.voxel_map.xy_to_grid_coords(xy)  # type: ignore
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
        xy = self.space.voxel_map.grid_coords_to_xy(pt)  # type: ignore
        return float(xy[0]), float(xy[1])

    def compute_dis(self, a: Tuple[int, int], b: Tuple[int, int]):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def compute_obstacle_punishment(self, a: Tuple[int, int], weight: int, avoid: int) -> float:
        obstacle_punishment = 0
        # it is a trick, if we compute euclidean dis between a and every obstacle,
        # this single compute_obstacle_punishment will be O(n) complexity
        # so we just check a square of size (2*avoid) * (2*avoid)
        # centered at a, which is O(1) complexity
        for i in range(-avoid, avoid + 1):
            for j in range(-avoid, avoid + 1):
                if self.point_is_occupied(a[0] + i, a[1] + j):
                    b = [a[0] + i, a[1] + j]
                    obs_dis = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                    obstacle_punishment = max((weight / max(obs_dis, 1)), obstacle_punishment)
        return obstacle_punishment

    # A* heuristic
    def compute_heuristic(self, a: Tuple[int, int], b: Tuple[int, int], weight=6, avoid=3) -> float:
        return (
            self.compute_dis(a, b)
            + weight * self.compute_obstacle_punishment(a, weight, avoid)
            + self.compute_obstacle_punishment(b, weight, avoid)
        )

    def is_in_line_of_sight(self, start_pt: Tuple[int, int], end_pt: Tuple[int, int]) -> bool:
        """Checks if there is a line-of-sight between two points.

        Implements using Bresenham's line algorithm.

        Args:
            start_pt: The starting point.
            end_pt: The ending point.

        Returns:
            Whether there is a line-of-sight between the two points.
        """

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if self.point_is_occupied(x, y):
                        return False

        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if self.point_is_occupied(x, y):
                        return False

        return True

    def is_a_line(self, a, b, c):
        if a[0] == b[0]:
            return c[0] == a[0]
        if b[0] == c[0]:
            return False
        return ((c[1] - b[1]) / (c[0] - b[0])) == ((b[1] - a[1]) / (b[0] - a[0]))

    def clean_path_for_xy(self, waypoints):
        goal = waypoints[-1]
        waypts = [self.to_pt(waypoint) for waypoint in waypoints]
        waypts = self.clean_path(waypts)
        waypoints = [self.to_xy(waypt) for waypt in waypts]
        traj = []
        for i in range(len(waypoints) - 1):
            theta = self.compute_theta(
                waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1]
            )
            traj.append([waypoints[i][0], waypoints[i][1], float(theta)])
        traj.append([waypoints[-1][0], waypoints[-1][1], goal[-1]])
        return traj

    def clean_path(self, path) -> List[Tuple[int, int]]:
        """Cleans up the final path.

        This implements a simple algorithm where, given some current position,
        we find the last point in the path that is in line-of-sight, and then
        we set the current position to that point. This is repeated until we
        reach the end of the path. This is not particularly efficient, but
        it's simple and works well enough.

        Args:
            path: The path to clean up.

        Returns:
            The cleaned up path.
        """
        cleaned_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self.is_in_line_of_sight(path[i][:2], path[j][:2]):
                    break
            else:
                j = i + 1
            # Include the mid waypoint to avoid the collision
            # if j - i >= 2:
            #     cleaned_path.append(path[(i + j) // 2])
            cleaned_path.append(path[j])
            i = j
        return cleaned_path

    def get_unoccupied_neighbor(self, pt: Tuple[int, int], goal_pt=None) -> Tuple[int, int]:
        if not self.point_is_occupied(*pt):
            return pt

        # This is a sort of hack to deal with points that are close to an edge.
        # If the start point is occupied, we check adjacent points until we get
        # one which is unoccupied. If we can't find one, we throw an error.
        neighbor_pts = neighbors(pt)
        if goal_pt is not None:
            goal_pt_non_null = goal_pt
            neighbor_pts.sort(key=lambda n: self.compute_heuristic(n, goal_pt_non_null))
        for neighbor_pt in neighbor_pts:
            if not self.point_is_occupied(*neighbor_pt):
                return neighbor_pt
        print(
            "The robot might stand on a non navigable point, so check obstacle map and restart roslaunch"
        )
        return None
        # raise ValueError("The robot might stand on a non navigable point, so check obstacle map and restart roslaunch")

    def get_reachable_points(self, start_pt: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Gets all reachable points from a given starting point.

        Args:
            start_pt: The starting point

        Returns:
            The set of all reachable points
        """

        self.reset()
        start_pt = self.get_unoccupied_neighbor(start_pt)
        if start_pt is None:
            return set()

        reachable_points: set[Tuple[int, int]] = set()
        to_visit = [start_pt]
        while to_visit:
            pt = to_visit.pop()
            if pt in reachable_points:
                continue
            reachable_points.add(pt)
            for new_pt in neighbors(pt):
                if new_pt in reachable_points:
                    continue
                if self.point_is_occupied(new_pt[0], new_pt[1]):
                    continue
                to_visit.append(new_pt)
        return reachable_points

    def run_astar(
        self,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        remove_line_of_sight_points: bool = False,
    ) -> List[Tuple[float, float]]:

        start_pt, end_pt = self.to_pt(start_xy), self.to_pt(end_xy)

        # Checks that both points are unoccupied.
        # start_pt = self.get_unoccupied_neighbor(start_pt, end_pt)
        start_pt = self.get_unoccupied_neighbor(start_pt)
        end_pt = self.get_unoccupied_neighbor(end_pt, start_pt)
        # print('A* formally starts ', time.time() - self.start_time, ' seconds after path planning starts')
        if start_pt is None or end_pt is None:
            return None

        # Implements A* search.
        q = [(0, start_pt)]
        came_from: dict = {start_pt: None}
        cost_so_far: dict[Tuple[int, int], float] = {start_pt: 0.0}
        while q:
            _, current = heapq.heappop(q)

            if current == end_pt:
                break

            for nxt in neighbors(current):
                if self.point_is_occupied(nxt[0], nxt[1]):
                    continue
                new_cost = cost_so_far[current] + self.compute_heuristic(current, nxt)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.compute_heuristic(end_pt, nxt)
                    heapq.heappush(q, (priority, nxt))  # type: ignore
                    came_from[nxt] = current

        else:
            return None

        # Reconstructs the path.
        path = []
        current = end_pt
        while current != start_pt:
            path.append(current)
            prev = came_from[current]
            if prev is None:
                break
            current = prev
        path.append(start_pt)
        path.reverse()

        # Clean up the path.
        if remove_line_of_sight_points:
            path = self.clean_path(path)

        # return [start_xy] + [self.to_xy(pt) for pt in path[1:-1]] + [end_xy]
        return [start_xy] + [self.to_xy(pt) for pt in path[1:]]

    def plan(self, start, goal, verbose: bool = True) -> PlanResult:
        """plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py
        """
        # assert len(start) == self.space.dof, "invalid start dimensions"
        # assert len(goal) == self.space.dof, "invalid goal dimensions"
        # self.start_time = time.time()
        self.reset()
        # if not self.space.is_valid(goal):
        #     if verbose:
        #         print("[Planner] invalid goal")
        #     return PlanResult(False, reason="[Planner] invalid goal")
        # Add start to the tree
        # print('Start running A* ', time.time() - self.start_time, ' seconds after path planning starts')
        waypoints = self.run_astar(start[:2], goal[:2])
        # print('Finish running A* ', time.time() - self.start_time, ' seconds after path planning starts')

        if waypoints is None:
            if verbose:
                print("A* fails, check obstacle map")
            return PlanResult(False, reason="A* fails, check obstacle map")
        trajectory = []
        for i in range(len(waypoints) - 1):
            theta = self.compute_theta(
                waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1]
            )
            trajectory.append(Node([waypoints[i][0], waypoints[i][1], float(theta)]))
        trajectory.append(Node([waypoints[-1][0], waypoints[-1][1], goal[-1]]))
        # print('Finish computing theta ', time.time() - self.start_time, ' seconds after path planning starts')
        return PlanResult(True, trajectory=trajectory)
