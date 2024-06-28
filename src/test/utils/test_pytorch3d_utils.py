from stretch.utils.pytorch3d_helpers import box3d_overlap

import torch

def _make_box(l1, l2, l3):
    """l: lengths of sides"""
    box = torch.zeros([1, 8, 3])
    box[0, 0, :] = torch.tensor([1., -1., 1.])
    box[0, 1, :] = torch.tensor([1., 1., 1.])
    box[0, 2, :] = torch.tensor([1., 1., -1.])
    box[0, 3, :] = torch.tensor([1., -1., -1.])
    box[0, 4, :] = torch.tensor([-1., -1., 1.])
    box[0, 5, :] = torch.tensor([-1., 1., 1.])
    box[0, 6, :] = torch.tensor([-1., 1., -1.])
    box[0, 7, :] = torch.tensor([-1., -1., -1.])

    box[0, :, 0] *= l1
    box[0, :, 1] *= l2
    box[0, :, 2] *= l3

    return box

def _make_boxes(N, M):
    b1 = torch.zeros([N, 8, 3])
    b2 = torch.zeros([M, 8, 3])

    for i in range(N):
        x = 0.5 + 0.1*i
        b1[i, :, :] = _make_box(x, x ,x)

    for i in range(M):
        x = 0.5 + 0.1*i
        b2[i, :, :] = _make_box(x, x ,x)

    return b1, b2

def test_box3d_overlap():
    N = 2
    M = 3

    # box: n (#) x 8 (corners) x 3 (XYZ)
    boxes1, boxes2 = _make_boxes(N, M)
    vol, iou = box3d_overlap(boxes1, boxes2)
    print(boxes1)
    print(boxes2)
    print("VoI:", vol)
    print("IoU:", iou)

def _test_all():
    test_box3d_overlap()

if __name__ == '__main__':
    _test_all()