"""
PointNet++ CUDA operations for efficient point cloud processing.

This module provides GPU-accelerated operations for PointNet++.
"""

from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from opencood.pcdet_utils.pointnet2.pointnet2_batch import pointnet2_batch_cuda as pointnet2  # type: ignore[attr-defined]


class GroupingOperation(Function):
    """
    Group features by indices for local feature aggregation.

    CUDA-accelerated grouping with custom backward for batched inputs.
    """

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Group features according to neighborhood indices.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving backward information.
        features : torch.Tensor
            Input features with shape (B, C, N).
        idx : torch.Tensor
            Grouping indices with shape (B, npoint, nsample).
            Values in range [0, N-1].

        Returns
        -------
        output : torch.Tensor
            Grouped features with shape (B, C, npoint, nsample).
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.empty((B, C, nfeatures, nsample), device=features.device, dtype=features.dtype)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Compute gradient with respect to input features.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context with saved forward information.
        grad_out : torch.Tensor
            Output gradient with shape (B, C, npoint, nsample).

        Returns
        -------
        grad_features : torch.Tensor
            Input features gradient with shape (B, C, N).
        None
            Placeholder for idx gradient.
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros((B, C, N), device=grad_out.device, dtype=grad_out.dtype)

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    """
    Ball query operation for finding neighbors within a radius.

    CUDA-accelerated local neighborhood query for batched inputs.
    """

    @staticmethod
    def forward(ctx: Any, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        Find neighbors within radius for each query point.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for backward (unused).
        radius : float
            Search radius for ball query.
        nsample : int
            Maximum number of neighbors to sample per ball.
        xyz : torch.Tensor
            Point coordinates with shape (B, N, 3).
        new_xyz : torch.Tensor
            Query point coordinates with shape (B, npoint, 3).

        Returns
        -------
        idx : torch.Tensor
            Neighbor indices with shape (B, npoint, nsample).
            Values in range [0, N-1].
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.zeros((B, npoint, nsample), device=xyz.device, dtype=torch.int32)

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx: Any, a: Optional[torch.Tensor] = None) -> Tuple[None, None, None, None]:
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    """
    Query and group points within a ball radius.

    Combines ball query and grouping operations for local feature extraction
    in batched point clouds.

    Parameters
    ----------
    radius : float
        Ball radius for neighborhood query.
    nsample : int
        Maximum number of points to sample per ball.
    use_xyz : bool, optional
        If True, concatenates relative xyz coordinates to features.
        Default is True.

    Attributes
    ----------
    radius : float
        Stored ball radius.
    nsample : int
        Stored maximum sample count.
    use_xyz : bool
        Whether to concatenate xyz coordinates.
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Query neighbors and group their features.

        Parameters
        ----------
        xyz : torch.Tensor
            Point coordinates with shape (B, N, 3).
        new_xyz : torch.Tensor
            Query point coordinates with shape (B, npoint, 3).
        features : torch.Tensor or None, optional
            Point features with shape (B, C, N). Default is None.

        Returns
        -------
        new_features : torch.Tensor
            Grouped features with shape (B, C_out, npoint, nsample).
            If use_xyz=True: C_out = C + 3, else C_out = C.
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features
