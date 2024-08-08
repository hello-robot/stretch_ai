# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import torch

from stretch.utils.pytorch3d import box3d_overlap
from stretch.utils.pytorch3d.pytorch3d_helpers import _box3d_overlap


def _make_box(l1, l2, l3):
    """l: lengths of sides"""
    box = torch.zeros([1, 8, 3])
    box[0, 0, :] = torch.tensor([1.0, -1.0, 1.0])
    box[0, 1, :] = torch.tensor([1.0, 1.0, 1.0])
    box[0, 2, :] = torch.tensor([1.0, 1.0, -1.0])
    box[0, 3, :] = torch.tensor([1.0, -1.0, -1.0])
    box[0, 4, :] = torch.tensor([-1.0, -1.0, 1.0])
    box[0, 5, :] = torch.tensor([-1.0, 1.0, 1.0])
    box[0, 6, :] = torch.tensor([-1.0, 1.0, -1.0])
    box[0, 7, :] = torch.tensor([-1.0, -1.0, -1.0])

    box[0, :, 0] *= l1 / 2.0
    box[0, :, 1] *= l2 / 2.0
    box[0, :, 2] *= l3 / 2.0

    return box


def _make_boxes(N, M, x_min=1.0):
    b1 = torch.zeros([N, 8, 3])
    b2 = torch.zeros([M, 8, 3])

    for i in range(N):
        x = x_min + 0.1 * i
        b1[i, :, :] = _make_box(x, x, x)

    for i in range(M):
        x = x_min + 0.1 * i
        b2[i, :, :] = _make_box(x, x, x)

    return b1, b2


def test_box3d_overlap():
    N = 2
    M = 3

    # check VoI / IOU
    # box: n (#) x 8 (corners) x 3 (XYZ)
    boxes1, boxes2 = _make_boxes(N, M, x_min=1.0)
    vol, iou = box3d_overlap(boxes1, boxes2)

    print("VoI:", vol)
    print("IoU:", iou)
    assert vol.shape == torch.Size([2, 3])
    assert iou.shape == torch.Size([2, 3])

    boxes1, _ = _make_boxes(1, 1, x_min=2.0)
    box_volume = _box3d_overlap.volume(boxes1)
    assert box_volume == torch.tensor(8.0)


def _test_all():
    test_box3d_overlap()


if __name__ == "__main__":
    _test_all()
