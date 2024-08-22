# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) 2023 PyG Team <team@pyg.org>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the parent directory of this file.

import itertools
import numbers
import typing
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

USE_TORCH_CLUSTER = False
if USE_TORCH_CLUSTER:
    try:
        from torch_cluster import grid_cluster
    except ImportError:
        print("torch_cluster not found. Using custom implementation.")
        grid_cluster = None
if not USE_TORCH_CLUSTER or grid_cluster is None:
    from stretch.utils.torch_cluster import grid_cluster


def consecutive_cluster(src):
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def repeat(src: Any, length: int) -> Any:
    if src is None:
        return None

    if isinstance(src, Tensor):
        if src.numel() == 1:
            return src.repeat(length)

        if src.numel() > length:
            return src[:length]

        if src.numel() < length:
            last_elem = src[-1].unsqueeze(0)
            padding = last_elem.repeat(length - src.numel())
            return torch.cat([src, padding])

        return src

    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))

    if len(src) > length:
        return src[:length]

    if len(src) < length:
        return src + list(itertools.repeat(src[-1], length - len(src)))

    return src


@typing.no_type_check
def voxel_grid(
    pos: Tensor,
    size: Union[float, List[float], Tensor],
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, List[float], Tensor]] = None,
    end: Optional[Union[float, List[float], Tensor]] = None,
) -> Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (torch.Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    if grid_cluster is None:
        print("Warning: `torch_cluster` not found. Using custom implementation.")
        return voxel_grid_no_cluster(pos, size, batch, start, end)

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    dim = pos.size(1)

    if batch is None:
        batch = pos.new_zeros(pos.size(0), dtype=torch.long)

    pos = torch.cat([pos, batch.view(-1, 1).to(pos.dtype)], dim=-1)

    if not isinstance(size, Tensor):
        size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    size = repeat(size, dim)
    size = torch.cat([size, size.new_ones(1)])  # Add additional batch dim.

    if start is not None:
        if not isinstance(start, Tensor):
            start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
        start = repeat(start, dim)
        start = torch.cat([start, start.new_zeros(1)])

    if end is not None:
        if not isinstance(end, Tensor):
            end = torch.tensor(end, dtype=pos.dtype, device=pos.device)
        end = repeat(end, dim)
        end = torch.cat([end, batch.max().unsqueeze(0)])

    return grid_cluster(pos, size, start, end)


@typing.no_type_check
def voxel_grid_no_cluster(
    pos: Tensor,
    size: Union[float, List[float], Tensor],
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, List[float], Tensor]] = None,
    end: Optional[Union[float, List[float], Tensor]] = None,
) -> Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (torch.Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor`
    """
    # Ensure pos is a 2D tensor
    if pos.dim() != 2:
        raise ValueError("pos must be a 2D tensor")

    # Get number of points and dimensions
    num_points, num_dims = pos.size()

    # Convert size to tensor if it's not already
    if not isinstance(size, Tensor):
        size = torch.tensor(size, device=pos.device, dtype=pos.dtype)
    if size.dim() == 0:
        size = size.repeat(num_dims)

    # Determine start and end if not provided
    if start is None:
        start = pos.min(dim=0)[0]
    else:
        start = torch.tensor(start, device=pos.device, dtype=pos.dtype)

    if end is None:
        end = pos.max(dim=0)[0]
    else:
        end = torch.tensor(end, device=pos.device, dtype=pos.dtype)

    # Compute the number of voxels in each dimension
    num_voxels = ((end - start) / size).ceil().long()

    # Compute voxel indices for each point
    voxel_indices = ((pos - start) / size).long()

    # Clamp indices to ensure they're within bounds
    voxel_indices = torch.clamp(voxel_indices, min=torch.zeros_like(num_voxels), max=num_voxels - 1)

    # Compute unique voxel identifier for each point
    voxel_id = voxel_indices[:, 0]
    for i in range(1, num_dims):
        voxel_id = voxel_id * num_voxels[i] + voxel_indices[:, i]

    # If batch is provided, incorporate it into the voxel_id
    if batch is not None:
        batch_offset = batch * num_voxels.prod()
        voxel_id = voxel_id + batch_offset

    # Return unique voxel identifiers
    return voxel_id
