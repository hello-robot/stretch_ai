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
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .space import ConfigurationSpace, Node

"""
This just defines the standard interface for a motion planner
"""


class PlanResult(object):
    """Stores motion plan. Can be extended."""

    def __init__(
        self,
        success,
        trajectory: Optional[List] = None,
        reason: Optional[str] = None,
        planner: Optional["Planner"] = None,
    ):
        self.success = success
        self.trajectory = trajectory
        self.reason = reason
        self.planner = planner

    def get_success(self):
        """Was the trajectory planning successful?"""
        return self.success

    def get_trajectory(self, *args, **kwargs) -> Optional[List]:
        """Return the trajectory"""
        return self.trajectory

    def get_length(self):
        """Length of a plan"""
        if not self.success:
            return 0
        return len(self.trajectory)


class Planner(ABC):
    """planner base class"""

    def __init__(self, space: ConfigurationSpace, validate_fn: Callable):
        self._space = space
        self._validate = validate_fn
        self._nodes: Optional[List[Node]] = None

    @property
    def space(self) -> ConfigurationSpace:
        return self._space

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[Node]):
        self._nodes = nodes

    @abstractmethod
    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """returns a trajectory"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """reset the planner"""
        raise NotImplementedError

    def validate(self, state) -> bool:
        """Check if state is valid"""
        return self._validate(state)
