"""
3D IoU calculation and rotated NMS operations.

Provides CUDA-accelerated functions for computing 3D intersection over union (IoU),
generalized IoU (GIoU), and non-maximum suppression (NMS) for 3D bounding boxes.

Author: Shaoshuai Shi
All Rights Reserved 2019-2020.
"""

import torch

from typing import Tuple, Union, Optional

from opencood.pcdet_utils.iou3d_nms import iou3d_nms_cuda


def aligned_boxes_iou3d_gpu(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    return_union: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute 3D IoU for aligned box pairs (element-wise).

    Parameters
    ----------
    boxes_a : torch.Tensor
        First set of boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    boxes_b : torch.Tensor
        Second set of boxes with shape (N, 7).
    return_union : bool, optional
        If True, returns (iou, union). Default is False.

    Returns
    -------
    iou3d : torch.Tensor
        IoU values with shape (N, 1).
    union : torch.Tensor, optional
        Union volumes with shape (N, 1). Only if return_union=True.
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    assert boxes_a.shape[0] == boxes_b.shape[0]
    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)
    overlaps_bev = torch.diagonal(overlaps_bev).reshape(-1, 1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)
    union = torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    iou3d = overlaps_3d / union
    if return_union:
        return iou3d, union
    return iou3d


def boxes_iou3d_gpu(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    return_union: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute 3D IoU between two sets of boxes on GPU.

    Parameters
    ----------
    boxes_a : torch.Tensor
        First set of boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    boxes_b : torch.Tensor
        Second set of boxes with shape (M, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    return_union : bool, optional
        If True, returns (iou, union). Default is False.

    Returns
    -------
    iou3d : torch.Tensor
        IoU matrix with shape (N, M).
    union : torch.Tensor, optional
        Union volumes with shape (N, M).
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    union = torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    iou3d = overlaps_3d / union
    if return_union:
        return iou3d, union
    return iou3d


def nms_gpu(boxes: torch.Tensor, scores: torch.Tensor, thresh: float, pre_maxsize: Optional[int] = None, **kwargs) -> Tuple[torch.Tensor, None]:
    """
    Perform rotated NMS on GPU for BEV boxes.

    Parameters
    ----------
    boxes : torch.Tensor
        Boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    scores : torch.Tensor
        Confidence scores with shape (N,).
    thresh : float
        IoU threshold for suppression.
    pre_maxsize : int or None, optional
        Maximum boxes to consider before NMS. Default is None.
    **kwargs
        Additional unused arguments for compatibility.

    Returns
    -------
    keep : torch.Tensor
        Indices of kept boxes after NMS.
    None
        Placeholder for compatibility.
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes: torch.Tensor, scores: torch.Tensor, thresh: float, **kwargs) -> Tuple[torch.Tensor, None]:
    """
    Perform axis-aligned NMS on GPU (ignores heading).

    Parameters
    ----------
    boxes : torch.Tensor
        Boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    scores : torch.Tensor
        Confidence scores with shape (N,).
    thresh : float
        IoU threshold for suppression.
    **kwargs
        Additional unused arguments for compatibility.

    Returns
    -------
    keep : torch.Tensor
        Indices of kept boxes after NMS.
    None
        Placeholder for compatibility.
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None
