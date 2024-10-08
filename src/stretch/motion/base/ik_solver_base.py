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


from typing import List, Optional, Tuple, Union

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

    def compute_fk(
        self, q, link_name=None, ignore_missing_joints=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """given joint values return end-effector position and quaternion associated with it"""
        raise NotImplementedError()

    def get_frame_pose(
        self, q: Union[np.ndarray, List[float], dict], node_a: str, node_b: str
    ) -> np.ndarray:
        """Given joint values, return the pose of the frame attached to node_a in the frame of node_b.

        Args:
            q: joint values
            node_a: name of the node where the frame is attached
            node_b: name of the node in whose frame the pose is desired

        Returns:
            4x4 np.ndarray: the pose of the frame attached to node_a in the frame of node_b
        """
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
        custom_ee_frame: Optional[str] = None,
    ) -> Tuple[np.ndarray, bool, dict]:

        """
        Given an end-effector position and quaternion, return the joint states and a success flag.
        Some solvers (e.g. the PositionIKOptimizer solver) will return a result regardless; the success flag indicates
        if the solution is within the solver's expected error margins.
        The output dictionary can contain any helpful debugging information for the solver, to analyze (in a more
        method-specific way) how well a fit was found.
        """
        raise NotImplementedError()
