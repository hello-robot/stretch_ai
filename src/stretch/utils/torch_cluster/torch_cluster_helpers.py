# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the parent directory of this file.

from typing import Optional

import torch


def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NOTE: modified by Hello Robot, Inc., to replace a call to torch.ops

    A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        pos (Tensor): D-dimensional position of points.
        size (Tensor): Size of a voxel in each dimension.
        start (Tensor, optional): Start position of the grid (in each
            dimension). (default: :obj:`None`)
        end (Tensor, optional): End position of the grid (in each
            dimension). (default: :obj:`None`)
    """
    # Ensure pos is a 2D tensor
    if pos.dim() != 2:
        raise ValueError("pos must be a 2D tensor")

    # Get number of points and dimensions
    num_points, num_dims = pos.size()

    # If start or end are not provided, compute them from pos
    if start is None:
        start = pos.min(dim=0)[0]
    if end is None:
        end = pos.max(dim=0)[0]

    # Ensure start, end, and size have the correct shape
    start = start.to(pos.device).to(pos.dtype)
    end = end.to(pos.device).to(pos.dtype)
    size = size.to(pos.device).to(pos.dtype)

    if start.dim() == 0:
        start = start.repeat(num_dims)
    if end.dim() == 0:
        end = end.repeat(num_dims)
    if size.dim() == 0:
        size = size.repeat(num_dims)

    # Compute the number of voxels in each dimension
    num_voxels = ((end - start) / size).ceil().long()

    # Num voxels must be at least 1 in each dimension
    num_voxels = torch.max(num_voxels, torch.ones_like(num_voxels))

    # Compute voxel indices for each point
    voxel_indices = ((pos - start) / size).round().long()

    # Clamp indices to ensure they're within bounds
    voxel_indices = torch.clamp(voxel_indices, min=torch.zeros_like(num_voxels), max=num_voxels - 1)

    # Compute unique voxel identifier for each point
    voxel_id = voxel_indices[:, 0]
    for i in range(1, num_dims):
        voxel_id = voxel_id * num_voxels[i] + voxel_indices[:, i]

    return voxel_id
