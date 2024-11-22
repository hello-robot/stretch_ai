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

# Based on Caelan Garrett's code from here: https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py


import time
from random import random
from typing import Callable, List, Optional, Tuple

import numpy as np

from stretch.motion.algo.node import TreeNode
from stretch.motion.base import ConfigurationSpace, Planner, PlanResult


class RRT(Planner):
    """Define RRT planning problem and parameters"""

    def __init__(
        self,
        space: ConfigurationSpace,
        validate_fn: Callable,
        p_sample_goal: float = 0.1,
        goal_tolerance: float = 1e-4,
        max_iter: int = 100,
    ):
        """Create RRT planner with configuration"""
        super(RRT, self).__init__(space, validate_fn)
        self.p_sample_goal = p_sample_goal
        self.goal_tolerance = goal_tolerance
        self.max_iter = max_iter
        self.reset()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @property
    def space(self):
        return self._space

    def reset(self):
        self.start_time = None
        self.goal_state = None
        self.nodes = []

    def plan(self, start, goal, verbose: bool = True, **kwargs) -> PlanResult:
        """plan from start to goal. creates a new tree.

        Based on Caelan Garrett's code (MIT licensed):
        https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py
        """
        assert len(start) == self.space.dof, "invalid start dimensions"
        assert len(goal) == self.space.dof, "invalid goal dimensions"
        self.reset()
        self.start_time = time.time()
        if not self.validate(start):
            if verbose:
                print("[Planner] invalid start")
            return PlanResult(False)
        if not self.validate(goal):
            if verbose:
                print("[Planner] invalid goal")
            return PlanResult(False)
        # Add start to the tree
        self.nodes.append(TreeNode(start))

        # TODO: currently not supporting goal samplers
        # if callable(goal):
        #    self.sample_goal = goal
        # else:
        #    # We'll assume this is valid
        #    self.sample_goal = lambda: goal
        self.goal_state = goal
        # Always try goal first
        res, _ = self.step_planner(force_sample_goal=True)
        if res.success:
            return res
        # Iterate a bunch of times
        for i in range(self.max_iter - 1):
            res, _ = self.step_planner(nodes=self.nodes)
            if res.success:
                return res
        return PlanResult(False)

    def step_planner(
        self,
        force_sample_goal=False,
        nodes: Optional[List[TreeNode]] = None,
        next_state: Optional[np.ndarray] = None,
    ) -> Tuple[PlanResult, TreeNode]:
        """Continue planning for a while. In case you want to try for anytime planning.

        Args:
            force_sample_goal: Whether to force sampling the goal
            nodes: The nodes to use
            next_state: The next state to try

        Returns:
            PlanResult: The result of the planning
            TreeNode: The last node in the tree
        """
        assert self.goal_state is not None, "no goal provided with a call to plan(start, goal)"
        assert (
            self.start_time is not None
        ), "does not look like you started planning with plan(start, goal)"

        if force_sample_goal or next_state is not None:
            should_sample_goal = True
        else:
            should_sample_goal = random() < self.p_sample_goal

        # If we do not pass in any nodes, use the planner's stored set of nodes
        if nodes is None:
            nodes = self.nodes

        # Get a new state
        if next_state is not None:
            goal_state = next_state
        else:
            goal_state = self.goal_state
        # Set the state we will try to move to
        if next_state is None:
            next_state = goal_state if should_sample_goal else self.space.sample()
        closest = self.space.closest_node_to_state(next_state, nodes)
        for step_state in self.space.extend(closest.state, next_state):
            if not self.validate(step_state):
                # This did not work
                break
            else:
                # Create a new TreeNode poining back to closest node
                closest = TreeNode(step_state, parent=closest)
                nodes.append(closest)
            # Check to see if it's the goal
            if self.space.distance(nodes[-1].state, goal_state) < self.goal_tolerance:
                # We made it! We're close enough to goal to be done
                return PlanResult(True, nodes[-1].backup()), nodes[-1]
        return PlanResult(False), closest
