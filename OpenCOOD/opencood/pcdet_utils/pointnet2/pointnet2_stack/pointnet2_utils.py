"""
PointNet++ CUDA operations for point cloud processing.

This module provides CUDA-accelerated operations for PointNet++ including
ball query, feature grouping, farthest point sampling, and feature interpolation.
All operations have custom autograd functions for efficient backpropagation.
"""

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from typing import Tuple

from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_stack_cuda as pointnet2


class BallQuery(Function):
    """
    Ball query operation for finding neighbors within a radius.
    
    CUDA-accelerated autograd function for local neighborhood queries.
    """

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor, new_xyz: torch.Tensor, new_xyz_batch_cnt):
        """
        Find neighbors within radius for each query point.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving backward information.
        radius : float
            Search radius for ball query.
        nsample : int
            Maximum number of neighbors to sample per ball.
        xyz : torch.Tensor
            Point coordinates.
        xyz_batch_cnt : torch.Tensor
            Point counts per batch.
        new_xyz : torch.Tensor
            Query point coordinates.
        new_xyz_batch_cnt : torch.Tensor
            Query counts per batch.
        Returns
        -------
        idx : torch.Tensor
            Neighbor indices with shape.
        empty_ball_mask : torch.Tensor
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = idx[:, 0] == -1
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):
    """
    Group features by indices for local feature aggregation.
    
    CUDA-accelerated autograd function with custom backward pass.
    """
    
    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor, idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Group features according to neighborhood indices.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving backward information.
        features : torch.Tensor
            Input features with shape (N1+N2+..., C).
        features_batch_cnt : torch.Tensor
            Feature counts per batch with shape (batch_size,).
            Format: [N1, N2, ...].
        idx : torch.Tensor
            Grouping indices with shape (M1+M2+..., nsample).
            Values in range [0, N1+N2+...-1].
        idx_batch_cnt : torch.Tensor
            Index counts per batch with shape (batch_size,).
            Format: [M1, M2, ...].

        Returns
        -------
        output : torch.Tensor
            Grouped features with shape (M1+M2+..., C, nsample).
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), "features: %s, features_batch_cnt: %s" % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), "idx: %s, idx_batch_cnt: %s" % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample)

        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
       Compute gradient with respect to input features.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context with saved forward information.
        grad_out : torch.Tensor
            Output gradient with shape (M1+M2+..., C, nsample).

        Returns
        -------
        grad_features : torch.Tensor
            Input features gradient with shape (N1+N2+..., C).
        None
            Placeholder for features_batch_cnt gradient.
        None
            Placeholder for idx gradient.
        None
            Placeholder for idx_batch_cnt gradient.
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx, idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    """
    Query and group points within a ball radius.
    
    Combines ball query and grouping operations for local feature extraction.
    Core building block for PointNet++ set abstraction layers.

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

    def forward(
        self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor, new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor, features: torch.Tensor = None
    ):
        """
        Query neighbors and group their features.

        Parameters
        ----------
        xyz : torch.Tensor
            Point coordinates with shape (N1+N2+..., 3).
        xyz_batch_cnt : torch.Tensor
            Point counts per batch with shape (batch_size,).
            Format: [N1, N2, ...].
        new_xyz : torch.Tensor
            Query point coordinates with shape (M1+M2+..., 3).
        new_xyz_batch_cnt : torch.Tensor
            Query counts per batch with shape (batch_size,).
            Format: [M1, M2, ...].
        features : torch.Tensor or None, optional
            Point features with shape (N1+N2+..., C). Default is None.

        Returns
        -------
        new_features : torch.Tensor
            Grouped features with shape (M1+M2+..., C_out, nsample).
            If use_xyz=True: C_out = C + 3, else C_out = C.
        idx : torch.Tensor
            Neighbor indices with shape (M1+M2+..., nsample).
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), "xyz: %s, xyz_batch_cnt: %s" % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), "new_xyz: %s, new_xyz_batch_cnt: %s" % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class FurthestPointSampling(Function):
    """
    Furthest point sampling for downsampling point clouds.
    
    CUDA-accelerated iterative farthest point selection algorithm.
    Selects diverse subset of points by maximizing minimum distances.
    """

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Sample points by iteratively selecting farthest point.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for backward (unused).
        xyz : torch.Tensor
            Point coordinates with shape (B, N, 3) where N > npoint.
        npoint : int
            Number of points to sample.

        Returns
        -------
        output : torch.Tensor
            Sampled point indices with shape (B, npoint).
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


class ThreeNN(Function):
    """
    Find three nearest neighbors for each query point.
    
    CUDA-accelerated k-NN search with k=3 for feature interpolation.
    """

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, unknown_batch_cnt: torch.Tensor, known: torch.Tensor, known_batch_cnt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find 3 nearest neighbors in known set for each unknown point.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for backward (unused).
        unknown : torch.Tensor
            Query point coordinates with shape (N1+N2+..., 3).
        unknown_batch_cnt : torch.Tensor
            Query counts per batch with shape (batch_size,).
        known : torch.Tensor
            Reference point coordinates with shape (M1+M2+..., 3).
        known_batch_cnt : torch.Tensor
            Reference counts per batch with shape (batch_size,).

        Returns
        -------
        dist : torch.Tensor
            L2 distances to 3 nearest neighbors with shape (N1+N2+..., 3).
        idx : torch.Tensor
            Indices of 3 nearest neighbors with shape (N1+N2+..., 3).
        """
        assert unknown.shape.__len__() == 2 and unknown.shape[1] == 3
        assert known.shape.__len__() == 2 and known.shape[1] == 3
        assert unknown_batch_cnt.__len__() == known_batch_cnt.__len__()

        dist2 = unknown.new_zeros(unknown.shape)
        idx = unknown_batch_cnt.new_zeros(unknown.shape).int()

        pointnet2.three_nn_wrapper(unknown.contiguous(), unknown_batch_cnt.contiguous(), known.contiguous(), known_batch_cnt.contiguous(), dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    """
    Interpolate features using inverse distance weighting from 3 neighbors.
    
    CUDA-accelerated feature upsampling with custom backward pass.
    Implements differentiable weighted sum based on neighbor distances.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        """
        Interpolate features using weighted sum of 3 neighbors.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context for saving backward information.
        features : torch.Tensor
            Source features with shape (M1+M2+..., C).
        idx : torch.Tensor
            Neighbor indices with shape (N1+N2+..., 3).
        weight : torch.Tensor
            Interpolation weights with shape (N1+N2+..., 3).
            Typically inverse distance weights (sum to 1).

        Returns
        -------
        output : torch.Tensor
            Interpolated features with shape (N1+N2+..., C).
        """
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        pointnet2.three_interpolate_wrapper(features.contiguous(), idx.contiguous(), weight.contiguous(), output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Compute gradient with respect to source features.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context with saved forward information.
        grad_out : torch.Tensor
            Output gradient with shape (N1+N2+..., C).

        Returns
        -------
        grad_features : torch.Tensor
            Source features gradient with shape (M1+M2+..., C).
        None
            Placeholder for idx gradient.
        None
            Placeholder for weight gradient.
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = grad_out.new_zeros((M, grad_out.shape[1]))
        pointnet2.three_interpolate_grad_wrapper(grad_out.contiguous(), idx.contiguous(), weight.contiguous(), grad_features)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


if __name__ == "__main__":
    pass
