"""
RoI-aware 3D pooling operations for point clouds with CUDA acceleration.

Provides GPU-accelerated pooling of point cloud features within 3D regions of
interest (RoIs) for object detection and segmentation tasks.
"""

from typing import Any, Union, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from opencood.utils import common_utils
from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_cuda


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
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices


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


class RoIAwarePool3d(nn.Module):
    """
    RoI-Aware 3D pooling module for point cloud features.

    Parameters
    ----------
    out_size : int or tuple
        Output size for pooled features (e.g., 7 or (7, 7, 7)).
    max_pts_each_voxel : int, optional
        Maximum number of points per voxel, by default 128.
    """

    def __init__(self, out_size: Union[int, Tuple[int, int, int]], max_pts_each_voxel: int = 128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois: torch.Tensor, pts: torch.Tensor, pts_feature: torch.Tensor, pool_method: str = "max") -> torch.Tensor:
        """
        Forward pass for RoI-aware pooling.

        Parameters
        ----------
        rois : torch.Tensor
            Regions of interest of shape (N, 7).
        pts : torch.Tensor
            Point coordinates of shape (npoints, 3).
        pts_feature : torch.Tensor
            Point features of shape (npoints, C).
        pool_method : str, optional
            Pooling method ('max' or 'avg'), by default 'max'.

        Returns
        -------
        torch.Tensor
            Pooled features of shape (N, out_x, out_y, out_z, C).

        Raises
        ------
        AssertionError
            If pool_method is not 'max' or 'avg'.
        """
        assert pool_method in ["max", "avg"]
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


class RoIAwarePool3dFunction(Function):
    """Custom autograd function for RoI-aware 3D pooling with CUDA acceleration."""

    @staticmethod
    def forward(ctx: Any, rois: torch.Tensor, pts: torch.Tensor, pts_feature: torch.Tensor, out_size: Union[int, Tuple[int, int, int]], max_pts_each_voxel: int, pool_method: str) -> torch.Tensor:
        """
        Forward pass for RoI-aware pooling.

        Parameters
        ----------
        ctx : Any
            Context object for saving variables for backward pass.
        rois : torch.Tensor
            Regions of interest of shape (N, 7) in format [x, y, z, dx, dy, dz, heading],
            where (x, y, z) is the box center.
        pts : torch.Tensor
            Point coordinates of shape (npoints, 3).
        pts_feature : torch.Tensor
            Point features of shape (npoints, C).
        out_size : int or tuple
            Output size (e.g., 7 or (7, 7, 7)).
        max_pts_each_voxel : int
            Maximum number of points per voxel.
        pool_method : str
            Pooling method ('max' or 'avg').

        Returns
        -------
        torch.Tensor
            Pooled features of shape (N, out_x, out_y, out_z, C).
        """
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)

        pool_method_map = {"max": 0, "avg": 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[None, None, torch.Tensor, None, None, None]:
        """
        Backward pass for RoI-aware pooling.

        Parameters
        ----------
        ctx : Any
            Context object with saved variables from forward pass.
        grad_out : torch.Tensor
            Gradient of output of shape (N, out_x, out_y, out_z, C).

        Returns
        -------
        Tuple[None, None, torch.Tensor, None, None, None]
            Gradients for all inputs. Only grad_in (for pts_feature) is computed,
            others are None.
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward

        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)

        return None, None, grad_in, None, None, None


if __name__ == "__main__":
    pass
