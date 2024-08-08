# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# This is how much memory we allocate
DEFAULT_GRID_SIZE = [1024, 1024]


class GridParams:
    """A 2d map that can be used for path planning. Maps in and out of the discrete grid."""

    def __init__(
        self,
        grid_size: Tuple[int, int],
        resolution: float,
        device: torch.device = torch.device("cpu"),
    ):

        if grid_size is not None:
            self.grid_size = [grid_size[0], grid_size[1]]
        else:
            self.grid_size = DEFAULT_GRID_SIZE

        # Track the center of the grid - (0, 0) in our coordinate system
        # We then just need to update everything when we want to track obstacles
        self.grid_origin = Tensor(self.grid_size + [0], device=device) // 2
        self.resolution = resolution
        # Used to track the offset from our observations so maps dont use too much space

        # Used for tensorized bounds checks
        self._grid_size_t = Tensor(self.grid_size, device=device)

    def xy_to_grid_coords(self, xy: torch.Tensor) -> Optional[np.ndarray]:
        """convert xy point to grid coords"""
        assert xy.shape[-1] == 2, "coords must be Nx2 or 2d array"
        # Handle conversion between world (X, Y) and grid coordinates
        if isinstance(xy, np.ndarray):
            xy = torch.from_numpy(xy).float()
        grid_xy = (xy / self.resolution) + self.grid_origin[:2]
        if torch.any(grid_xy >= self._grid_size_t) or torch.any(grid_xy < torch.zeros(2)):
            return None
        else:
            return grid_xy

    def grid_coords_to_xy(self, grid_coords: torch.Tensor) -> np.ndarray:
        """convert grid coordinate point to metric world xy point"""
        assert grid_coords.shape[-1] == 2, "grid coords must be an Nx2 or 2d array"
        return (grid_coords - self.grid_origin[:2]) * self.resolution

    def grid_coords_to_xyt(self, grid_coords: np.ndarray) -> np.ndarray:
        """convert grid coordinate point to metric world xyt point"""
        res = torch.zeros(3)
        res[:2] = self.grid_coords_to_xy(grid_coords)
        return res

    def mask_from_bounds(self, obstacles, explored, bounds: np.ndarray, debug: bool = False):
        """create mask from a set of 3d object bounds"""
        assert bounds.shape[0] == 3, "bounding boxes in xyz"
        assert bounds.shape[1] == 2, "min and max"
        assert (len(bounds.shape)) == 2, "only one bounding box"
        mins = torch.floor(self.xy_to_grid_coords(bounds[:2, 0])).long()
        maxs = torch.ceil(self.xy_to_grid_coords(bounds[:2, 1])).long()
        mask = torch.zeros_like(explored)
        mask[mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1] = True
        if debug:
            import matplotlib.pyplot as plt

            plt.imshow(obstacles.int() + explored.int() + mask.int())
        return mask
