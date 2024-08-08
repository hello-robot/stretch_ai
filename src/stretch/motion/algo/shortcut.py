# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from random import random
from typing import Callable, List

import numpy as np

from stretch.motion.algo.rrt import TreeNode
from stretch.motion.base import Planner, PlanResult


class Shortcut(Planner):
    """Define RRT planning problem and parameters. Holds two different trees and tries to connect them with some probabability."""

    def __init__(
        self,
        planner: Planner,
        shortcut_iter: int = 100,
    ):
        self.planner = planner
        super(Shortcut, self).__init__(self.planner.space, self.planner.validate)
        self.shortcut_iter = shortcut_iter
        self.reset()

    def reset(self):
        self.nodes = None

    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """Do shortcutting"""
        self.planner.reset()
        if verbose:
            print("Call internal planner")
        res = self.planner.plan(start, goal, verbose=verbose, **kwargs)
        self.nodes = self.planner.nodes
        if not res.success or len(res.trajectory) < 4:
            # Planning failed so nothing to do here
            return res
        # Now try to shorten things
        # print("Plan =")
        # for i, pt in enumerate(res.trajectory):
        #     print(i, pt.state)
        for i in range(self.shortcut_iter):
            # Sample two indices
            idx0 = np.random.randint(len(res.trajectory) - 3)
            idx1 = np.random.randint(idx0 + 1, len(res.trajectory))
            # print("connect", idx0, idx1)
            node_a = res.trajectory[idx0]
            node_b = res.trajectory[idx1]
            # Extend between them
            previous_node = node_a
            success = False
            for qi in self.space.extend(node_a.state, node_b.state):
                if np.all(qi == node_b.state):
                    success = True
                    break
                if not self.validate(qi):
                    break
                else:
                    self.nodes.append(TreeNode(qi, parent=previous_node))
                    previous_node = self.nodes[-1]
            else:
                success = True
            if success:
                # Finish by connecting the two
                # print("Connection success", idx1)
                node_b.parent = previous_node
        new_trajectory = res.trajectory[-1].backup()
        # print("Plan =")
        # for i, pt in enumerate(new_trajectory):
        #     print(i, pt.state)
        return PlanResult(True, new_trajectory, planner=self)
