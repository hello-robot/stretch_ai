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

import matplotlib.pyplot as plt
import numpy as np

from stretch.motion.base import ConfigurationSpace


class SimpleEnv:
    """Simple 2D environment for testing."""

    def __init__(self, obs: np.ndarray = None, size=10, obstacle_size=6, seed: int = 0):
        """Create simple 2d env. obs is obstacle location."""
        self.size = size
        self.obstacle_size = obstacle_size
        if obs is None:
            self.reset(seed)
        else:
            self.obstacle_pos = obs

    def get_space(self) -> ConfigurationSpace:
        """Get a space that we can use for planning"""
        return ConfigurationSpace(2, np.zeros(2), np.ones(2) * self.size, step_size=0.5)

    def validate(self, q: np.ndarray):
        """Check to see if this configuration is feasible for planning

        Args:
            q(np.ndarray): 2D numpy array denoting a point"""
        assert len(q) == 2
        x, y = q
        if x < 0 or y < 0 or x > self.size or y > self.size:
            # Out of bounds
            return False
        ox, oy = self.obstacle_pos
        if x > ox and y > oy and x < ox + self.obstacle_size and y < oy + self.obstacle_size:
            # inside the obstacle
            return False
        return True

    def reset(self, seed: int = 0):
        self.seed = seed
        np.random.seed(seed)
        self.obstacle_pos = np.random.random(2) * (self.size - self.obstacle_size)

    def show(self, states, backend: str = "mpl", show: bool = True):
        """Display the scene + states"""

        if backend != "mpl":
            raise NotImplementedError(f"Backend {backend} not yet supported.")

        # Create a figure and axis without axes
        fig, ax = plt.subplots()
        ax.axis("off")

        # Draw obstacle box
        obstacle_rect = plt.Rectangle(
            tuple(self.obstacle_pos),
            self.obstacle_size,
            self.obstacle_size,
            color="red",
            alpha=0.5,
        )
        ax.add_patch(obstacle_rect)

        # Plot trajectory points
        for traj_point in states:
            ax.plot(traj_point[0], traj_point[1], "bo")  # 'bo' means blue circles

        # Connect trajectory points with lines
        traj_x, traj_y = zip(*states)
        ax.plot(traj_x, traj_y, "b-")  # 'b-' means blue solid line

        if show:
            # Show the plot
            plt.show()

        return fig
