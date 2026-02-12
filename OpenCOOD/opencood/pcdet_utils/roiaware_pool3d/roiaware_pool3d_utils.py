"""
RoI-aware 3D pooling operations for point clouds with CUDA acceleration.

Provides GPU-accelerated pooling of point cloud features within 3D regions of
interest (RoIs) for object detection and segmentation tasks.
"""

import torch
from typing import cast

from opencood.utils import common_utils
from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_cuda  # type: ignore[attr-defined]


def points_in_boxes_cpu(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Find which points are inside which boxes using CPU computation.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates of shape (num_points, 3).
    boxes : torch.Tensor
        Boxes in format [x, y, z, dx, dy, dz, heading] of shape (N, 7),
        where (x, y, z) is the box center. Boxes DO NOT overlap.

    Returns
    -------
    torch.Tensor
        Point indices of shape (N, num_points) indicating which points are in each box.
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, points_is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, _ = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    if points_is_numpy:
        return cast(torch.Tensor, torch.from_numpy(point_indices.numpy()))
    return point_indices


def points_in_boxes_gpu(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Find which box each point belongs to using GPU computation.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates of shape (B, M, 3).
    boxes : torch.Tensor
        Boxes in format [x, y, z, dx, dy, dz, heading] of shape (B, T, 7),
        where num_valid_boxes <= T.

    Returns
    -------
    torch.Tensor
        Box indices for each point of shape (B, M). Background points are marked as -1.
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    return box_idxs_of_pts
