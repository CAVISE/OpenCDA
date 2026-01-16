"""
3D IoU calculation and rotated NMS operations.

Provides CUDA-accelerated functions for computing 3D intersection over union (IoU),
generalized IoU (GIoU), and non-maximum suppression (NMS) for 3D bounding boxes.

Author: Shaoshuai Shi
All Rights Reserved 2019-2020.
"""

import torch
import numpy as np
import numpy.typing as npt

from opencood.utils.common_utils import check_numpy_to_torch
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_cuda

from typing import Tuple, Optional, Union, List

def boxes_bev_iou_cpu(
    boxes_a: Union[torch.Tensor, npt.NDArray],
    boxes_b: Union[torch.Tensor, npt.NDArray],
) -> Union[torch.Tensor, npt.NDArray]:
    """
    Compute bird's eye view (BEV) IoU between two sets of boxes on CPU.

    Parameters
    ----------
    boxes_a : torch.Tensor or npt.NDArray
        First set of boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    boxes_b : torch.Tensor or npt.NDArray
        Second set of boxes with shape (M, 7).
        Format: [x, y, z, dx, dy, dz, heading].

    Returns
    -------
    ans_iou : torch.Tensor or npt.NDArray
        IoU matrix with shape (N, M).
        Returns same type as input (numpy or torch).

    """
    boxes_a, is_numpy = check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), "Only support CPU tensors"
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute bird's eye view (BEV) IoU between two sets of boxes on GPU.

    Parameters
    ----------
    boxes_a : torch.Tensor
        First set of boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    boxes_b : torch.Tensor
        Second set of boxes with shape (M, 7).
        Format: [x, y, z, dx, dy, dz, heading].

    Returns
    -------
    ans_iou : torch.Tensor
        IoU matrix with shape (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def decode_boxes_and_iou3d(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    pc_range: Union[List[float], torch.Tensor],
    box_mean: Union[List[float], torch.Tensor],
    box_std: Union[List[float], torch.Tensor],
) -> torch.Tensor:
    """
    Decode normalized boxes and compute 3D IoU.

    Parameters
    ----------
    boxes_a : torch.Tensor
        First set of normalized boxes with shape (N, 8).
        Format: [x_n, y_n, z_n, dx_n, dy_n, dz_n, sin_h, cos_h].
    boxes_b : torch.Tensor
        Second set of normalized boxes with shape (M, 8).
    pc_range : List[float] or torch.Tensor
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
    box_mean : List[float] or torch.Tensor
        Box normalization mean with length 8.
    box_std : List[float] or torch.Tensor
        Box normalization standard deviation with length 8.

    Returns
    -------
    iou : torch.Tensor
        3D IoU matrix with shape (N, M).
    """
    boxes_a_dec = decode_boxes(boxes_a, pc_range, box_mean, box_std)
    boxes_b_dec = decode_boxes(boxes_b, pc_range, box_mean, box_std)
    iou = boxes_iou3d_gpu(boxes_a_dec, boxes_b_dec)

    return iou


def decode_boxes(
    boxes: torch.Tensor,
    pc_range: Union[List[float], torch.Tensor],
    box_mean: Union[List[float], torch.Tensor],
    box_std: Union[List[float], torch.Tensor],
) -> torch.Tensor:
    """
    Decode normalized boxes to standard format.

    Parameters
    ----------
    boxes : torch.Tensor
        Normalized boxes with shape (N, 8).
        Format: [x_n, y_n, z_n, log(dx), log(dy), log(dz), sin_h, cos_h].
    pc_range : List[float] or torch.Tensor
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
    box_mean : List[float] or torch.Tensor
        Box normalization mean with length 8.
    box_std : List[float] or torch.Tensor
        Box normalization standard deviation with length 8.

    Returns
    -------
    boxes_out : torch.Tensor
        Decoded boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    """
    assert len(boxes.shape) == 2
    assert boxes.shape[1] == 8
    if isinstance(box_mean, list):
        box_mean = torch.tensor(box_mean, device=boxes.device)
    if isinstance(box_std, list):
        box_std = torch.tensor(box_std, device=boxes.device)
    boxes = boxes * box_std[None, :] + box_mean[None, :]
    boxes_out = torch.zeros((boxes.shape[0], 7), dtype=boxes.dtype, device=boxes.device)
    for i in range(3):
        boxes_out[:, i] = boxes[:, i] * (pc_range[i + 3] - pc_range[i]) + pc_range[i]
    boxes_out[:, 3:6] = boxes[:, 3:6].exp()
    boxes_out[:, 6] = torch.atan2(boxes[:, 6], boxes[:, 7])
    return boxes_out


def decode_boxes_and_giou3d(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    pc_range: Union[List[float], torch.Tensor],
    box_mean: Union[List[float], torch.Tensor],
    box_std: Union[List[float], torch.Tensor],
) -> torch.Tensor:
    """
    Decode normalized boxes and compute 3D Generalized IoU (GIoU).

    Parameters
    ----------
    boxes_a : torch.Tensor
        First set of normalized boxes with shape (N, 8).
    boxes_b : torch.Tensor
        Second set of normalized boxes with shape (M, 8).
    pc_range : List[float] or torch.Tensor
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
    box_mean : List[float] or torch.Tensor
        Box normalization mean with length 8.
    box_std : List[float] or torch.Tensor
        Box normalization standard deviation with length 8.

    Returns
    -------
    giou : torch.Tensor
        GIoU matrix with shape (N, M). Values in range [-1, 1].
    """
    boxes_a_dec = decode_boxes(boxes_a, pc_range, box_mean, box_std)
    boxes_b_dec = decode_boxes(boxes_b, pc_range, box_mean, box_std)
    corners_a = centroid_to_corners(boxes_a_dec)
    corners_b = centroid_to_corners(boxes_b_dec)
    iou, union = boxes_iou3d_gpu(boxes_a_dec, boxes_b_dec, return_union=True)
    lwh = torch.max(corners_a.max(dim=1)[0][:, None, :], corners_b.max(dim=1)[0]) - torch.min(
        corners_a.min(dim=1)[0][:, None, :], corners_b.min(dim=1)[0]
    )
    volume = lwh[..., 0] * lwh[..., 1] * lwh[..., 2]

    giou = iou - (volume - union) / volume

    return giou


def giou3d(boxes_a_dec: torch.Tensor, boxes_b_dec: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D Generalized IoU (GIoU) for decoded boxes.

    Parameters
    ----------
    boxes_a_dec : torch.Tensor
        First set of decoded boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].
    boxes_b_dec : torch.Tensor
        Second set of decoded boxes with shape (M, 7).

    Returns
    -------
    giou : torch.Tensor
        GIoU matrix with shape (N, M). 
    """
    corners_a = centroid_to_corners(boxes_a_dec)
    corners_b = centroid_to_corners(boxes_b_dec)
    iou, union = boxes_iou3d_gpu(boxes_a_dec, boxes_b_dec, return_union=True)
    lwh = torch.max(corners_a.max(dim=1)[0][:, None, :], corners_b.max(dim=1)[0]) - torch.min(
        corners_a.min(dim=1)[0][:, None, :], corners_b.min(dim=1)[0]
    )
    volume = lwh[..., 0] * lwh[..., 1] * lwh[..., 2]

    giou = iou - (volume - union) / volume

    return giou


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


def centroid_to_corners(
    boxes: Union[npt.NDArray, torch.Tensor]
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Convert boxes from centroid format to 8 corners.

    Parameters
    ----------
    boxes : npt.NDArray or torch.Tensor
        Boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].

    Returns
    -------
    corners : npt.NDArray or torch.Tensor
        Box corners with shape (N, 8, 3).
        Returns same type as input.
    """
    if isinstance(boxes, np.ndarray):
        corners = _centroid_to_corners_np(boxes)
    elif isinstance(boxes, torch.Tensor):
        corners = _centroid_to_corners_torch(boxes)
    else:
        raise TypeError("Input boxes should either be numpy array or torch tensor.")

    return corners


def _centroid_to_corners_torch(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from centroid format to corners (PyTorch).

    Parameters
    ----------
    boxes : torch.Tensor
        Boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].

    Returns
    -------
    corners : torch.Tensor
        Box corners with shape (N, 8, 3).
    """
    corners = torch.zeros((boxes.shape[0], 8, 3), dtype=boxes.dtype, device=boxes.device)
    sin_t = torch.sin(boxes[:, -1])
    cos_t = torch.cos(boxes[:, -1])
    corners[:, ::4, 0] = torch.stack([boxes[:, 0] + boxes[:, 3] / 2 * cos_t - boxes[:, 4] / 2 * sin_t] * 2, dim=1)  # lfx
    corners[:, ::4, 1] = torch.stack([boxes[:, 1] + boxes[:, 3] / 2 * sin_t + boxes[:, 4] / 2 * cos_t] * 2, dim=1)  # lfy
    corners[:, 1::4, 0] = torch.stack([boxes[:, 0] - boxes[:, 3] / 2 * cos_t - boxes[:, 4] / 2 * sin_t] * 2, dim=1)  # lbx
    corners[:, 1::4, 1] = torch.stack([boxes[:, 1] - boxes[:, 3] / 2 * sin_t + boxes[:, 4] / 2 * cos_t] * 2, dim=1)  # lby
    corners[:, 2::4, 0] = torch.stack([boxes[:, 0] - boxes[:, 3] / 2 * cos_t + boxes[:, 4] / 2 * sin_t] * 2, dim=1)  # rbx
    corners[:, 2::4, 1] = torch.stack([boxes[:, 1] - boxes[:, 3] / 2 * sin_t - boxes[:, 4] / 2 * cos_t] * 2, dim=1)  # rby
    corners[:, 3::4, 0] = torch.stack([boxes[:, 0] + boxes[:, 3] / 2 * cos_t + boxes[:, 4] / 2 * sin_t] * 2, dim=1)  # rfx
    corners[:, 3::4, 1] = torch.stack([boxes[:, 1] + boxes[:, 3] / 2 * sin_t - boxes[:, 4] / 2 * cos_t] * 2, dim=1)  # rfy
    corners[:, :, 2] = torch.cat(
        [torch.stack([boxes[:, 2] - boxes[:, 5] / 2] * 4, dim=1), torch.stack([boxes[:, 2] + boxes[:, 5] / 2] * 4, dim=1)], dim=1
    )

    return corners


def _centroid_to_corners_np(boxes: npt.NDArray) -> npt.NDArray:
    """
    Convert boxes from centroid format to corners (NumPy).

    Parameters
    ----------
    boxes : npt.NDArray
        Boxes with shape (N, 7).
        Format: [x, y, z, dx, dy, dz, heading].

    Returns
    -------
    corners : npt.NDArray
        Box corners with shape (N, 8, 3).
    """
    corners = np.zeros((boxes.shape[0], 8, 3), dtype=boxes.dtype)
    sin_t = np.sin(boxes[:, -1])
    cos_t = np.cos(boxes[:, -1])
    corners[:, ::4, 0] = np.stack([boxes[:, 0] + boxes[:, 3] / 2 * cos_t - boxes[:, 4] / 2 * sin_t] * 2, axis=1)  # lfx
    corners[:, ::4, 1] = np.stack([boxes[:, 1] + boxes[:, 3] / 2 * sin_t + boxes[:, 4] / 2 * cos_t] * 2, axis=1)  # lfy
    corners[:, 1::4, 0] = np.stack([boxes[:, 0] - boxes[:, 3] / 2 * cos_t - boxes[:, 4] / 2 * sin_t] * 2, axis=1)  # lbx
    corners[:, 1::4, 1] = np.stack([boxes[:, 1] - boxes[:, 3] / 2 * sin_t + boxes[:, 4] / 2 * cos_t] * 2, axis=1)  # lby
    corners[:, 2::4, 0] = np.stack([boxes[:, 0] - boxes[:, 3] / 2 * cos_t + boxes[:, 4] / 2 * sin_t] * 2, axis=1)  # rbx
    corners[:, 2::4, 1] = np.stack([boxes[:, 1] - boxes[:, 3] / 2 * sin_t - boxes[:, 4] / 2 * cos_t] * 2, axis=1)  # rby
    corners[:, 3::4, 0] = np.stack([boxes[:, 0] + boxes[:, 3] / 2 * cos_t + boxes[:, 4] / 2 * sin_t] * 2, axis=1)  # rfx
    corners[:, 3::4, 1] = np.stack([boxes[:, 1] + boxes[:, 3] / 2 * sin_t - boxes[:, 4] / 2 * cos_t] * 2, axis=1)  # rfy
    corners[:, :, 2] = np.concatenate(
        [np.stack([boxes[:, 2] - boxes[:, 5] / 2] * 4, axis=1), np.stack([boxes[:, 2] + boxes[:, 5] / 2] * 4, axis=1)], axis=1
    )

    return corners


def rotate_weighted_nms_gpu(
    box_preds,
    rbboxes,
    dir_labels,
    labels_preds,
    scores,
    iou_preds,
    anchors,
    pre_max_size=None,
    post_max_size=None,
    iou_threshold=0.5,
):
    """Original definition can be found in CIA_SSD paper"""
    if pre_max_size is not None:
        _ = scores.shape[0]  # num_keeped_scores


def nms_gpu(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    pre_maxsize: Optional[int] = None,
    **kwargs
) -> Tuple[torch.Tensor, None]:
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


def nms_normal_gpu(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    **kwargs
) -> Tuple[torch.Tensor, None]:
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
