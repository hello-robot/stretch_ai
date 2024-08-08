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


from typing import Optional, Tuple

import numpy as np


class IKSolverBase(object):
    """
    Base class for all IK solvers.
    """

    def get_dof(self) -> int:
        """returns dof for the manipulation chain"""
        raise NotImplementedError()

    def get_num_controllable_joints(self) -> int:
        """returns number of controllable joints under this solver's purview"""
        raise NotImplementedError()

    def compute_fk(self, q) -> Tuple[np.ndarray, np.ndarray]:
        """given joint values return end-effector position and quaternion associated with it"""
        raise NotImplementedError()

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray,
        q_init=None,
        max_iterations=100,
        num_attempts: int = 1,
        verbose: bool = False,
        ignore_missing_joints: bool = False,
    ) -> Tuple[np.ndarray, bool, dict]:

        """
        Given an end-effector position and quaternion, return the joint states and a success flag.
        Some solvers (e.g. the PositionIKOptimizer solver) will return a result regardless; the success flag indicates
        if the solution is within the solver's expected error margins.
        The output dictionary can contain any helpful debugging information for the solver, to analyze (in a more
        method-specific way) how well a fit was found.
        """
        raise NotImplementedError()
