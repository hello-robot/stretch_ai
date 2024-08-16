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
    pos = pos.view(pos.size(0), -1)

    start = torch.tensor([0.0])
    end = torch.tensor([0.0])

    pos = pos - start.unsqueeze(0)

    num_voxels = ((end - start) / size).to(torch.long) + 1
    num_voxels = num_voxels.cumprod(0)
    num_voxels = torch.cat(
        [torch.ones(1, dtype=num_voxels.dtype, device=num_voxels.device), num_voxels], 0
    )
    num_voxels = num_voxels.narrow(0, 0, size.size(0))

    out = (pos / size.view(1, -1)).to(torch.long)
    out *= num_voxels.view(1, -1)
    out = out.sum(1)

    return out
