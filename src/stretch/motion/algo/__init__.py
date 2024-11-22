# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Callable

from stretch.motion.base import ConfigurationSpace, Planner

from .a_star import AStar
from .rrt import RRT
from .rrt_connect import RRTConnect
from .shortcut import Shortcut
from .simplify import SimplifyXYT


def get_planner(algo: str, space: ConfigurationSpace, validate_fn: Callable, **kwargs) -> Planner:
    if algo == "rrt":
        return RRT(space, validate_fn, **kwargs)
    elif algo == "rrt_connect":
        return RRTConnect(space, validate_fn, **kwargs)
        return SimplifyXYT(**kwargs)
    elif algo == "a_star":
        return AStar(space, **kwargs)
    else:
        raise ValueError(f"Planner {algo} not supported. Choose from rrt, rrt_connect, a_star")
