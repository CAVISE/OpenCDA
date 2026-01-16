"""
PointNet++ CUDA operations for efficient point cloud processing.

This module provides GPU-accelerated operations for PointNet++.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from opencood.pcdet_utils.pointnet2.pointnet2_batch import pointnet2_batch_cuda as pointnet2


class FurthestPointSampling(Function):
    """
    Furthest point sampling for downsampling point clouds.
    
    CUDA-accelerated iterative farthest point selection for batched inputs.
    """

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Sample points by iteratively selecting farthest point.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for backward (unused).
        xyz : torch.Tensor
            Point coordinates with shape (B, N, 3) where N > npoint.
        npoint : int
            Number of points to sample per batch.

        Returns
        -------
        output : torch.Tensor
            Sampled point indices with shape (B, npoint).
            Values in range [0, N-1].
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    """
    Gather features by indices for batched point clouds.
    
    CUDA-accelerated feature gathering with custom backward pass.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Gather features according to indices.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving backward information.
        features : torch.Tensor
            Input features with shape (B, C, N).
        idx : torch.Tensor
            Gather indices with shape (B, npoint).
            Values in range [0, N-1].

        Returns
        -------
        output : torch.Tensor
            Gathered features with shape (B, C, npoint).
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Compute gradient with respect to input features.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context with saved forward information.
        grad_out : torch.Tensor
            Output gradient with shape (B, C, npoint).

        Returns
        -------
        grad_features : torch.Tensor
            Input features gradient with shape (B, C, N).
        None
            Placeholder for idx gradient.
        """
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    """
    Find three nearest neighbors for each query point.
    
    CUDA-accelerated k-NN search with k=3 for batched inputs.
    """

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find 3 nearest neighbors in known set for each unknown point.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for backward (unused).
        unknown : torch.Tensor
            Query point coordinates with shape (B, N, 3).
        known : torch.Tensor
            Reference point coordinates with shape (B, M, 3).

        Returns
        -------
        dist : torch.Tensor
            L2 distances to 3 nearest neighbors with shape (B, N, 3).
        idx : torch.Tensor
            Indices of 3 nearest neighbors with shape (B, N, 3).
            Values in range [0, M-1].
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    """
    Interpolate features using inverse distance weighting from 3 neighbors.
    
    CUDA-accelerated feature upsampling with custom backward for batched inputs.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Interpolate features using weighted sum of 3 neighbors.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving backward information.
        features : torch.Tensor
            Source features with shape (B, C, M).
        idx : torch.Tensor
            Neighbor indices with shape (B, n, 3).
        weight : torch.Tensor
            Interpolation weights with shape (B, n, 3).

        Returns
        -------
        output : torch.Tensor
            Interpolated features with shape (B, C, n).
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gradient with respect to source features.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context with saved forward information.
        grad_out : torch.Tensor
            Output gradient with shape (B, C, n).

        Returns
        -------
        grad_features : torch.Tensor
            Source features gradient with shape (B, C, M).
        None
            Placeholder for idx gradient.
        None
            Placeholder for weight gradient.
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    """
    Group features by indices for local feature aggregation.
    
    CUDA-accelerated grouping with custom backward for batched inputs.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
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
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

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
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
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
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
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

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
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


class GroupAll(nn.Module):
    """
    Group all points into a single global feature.
    
    Aggregates all points without spatial partitioning for global context.

    Parameters
    ----------
    use_xyz : bool, optional
        If True, concatenates xyz coordinates to features. Default is True.

    Attributes
    ----------
    use_xyz : bool
        Whether to concatenate xyz coordinates.
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        Group all points into single feature.

        Parameters
        ----------
        xyz : torch.Tensor
            Point coordinates with shape (B, N, 3).
        new_xyz : torch.Tensor
            Ignored (for interface compatibility).
        features : torch.Tensor or None, optional
            Point features with shape (B, C, N). Default is None.

        Returns
        -------
        new_features : torch.Tensor
            Grouped features with shape (B, C_out, 1, N).
            If use_xyz=True: C_out = C + 3, else C_out = C.
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
