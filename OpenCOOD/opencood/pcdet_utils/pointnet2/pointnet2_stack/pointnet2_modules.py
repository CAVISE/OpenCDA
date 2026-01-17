"""
PointNet++ modules for hierarchical point cloud feature learning.

This module provides set abstraction and feature propagation modules for
PointNet++ architecture, enabling multi-scale feature extraction and
upsampling in point cloud processing networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_utils

from typing import List, Optional, Tuple


class StackSAModuleMSG(nn.Module):
    """
    Multi-scale grouping (MSG) set abstraction module.

    Performs hierarchical feature learning at multiple scales using
    ball queries with different radii and MLP processing.

    Parameters
    ----------
    radii : List[float]
        List of ball query radii for each scale.
    nsamples : List[int]
        List of maximum sample counts per ball for each scale.
    mlps : List[List[int]]
        List of MLP specifications for each scale.
        Each inner list defines layer dimensions [C_in, C_1, ..., C_out].
    use_xyz : bool, optional
        If True, concatenates xyz coordinates to features. Default is True.
    pool_method : str, optional
        Pooling method: 'max_pool' or 'avg_pool'. Default is 'max_pool'.

    Attributes
    ----------
    groupers : nn.ModuleList
        List of QueryAndGroup modules for each scale.
    mlps : nn.ModuleList
        List of MLP networks for each scale.
    pool_method : str
        Stored pooling method.
    """

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method="max_pool"):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        xyz: torch.Tensor,
        xyz_batch_cnt: torch.Tensor,
        new_xyz: torch.Tensor,
        new_xyz_batch_cnt: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        empty_voxel_set_zeros: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-scale feature extraction with set abstraction.

        Parameters
        ----------
        xyz : torch.Tensor
            Input point coordinates with shape (N1+N2+..., 3).
        xyz_batch_cnt : torch.Tensor
            Input point counts per batch with shape (batch_size,).
            Format: [N1, N2, ...].
        new_xyz : torch.Tensor
            Sampled point coordinates with shape (M1+M2+..., 3).
        new_xyz_batch_cnt : torch.Tensor
            Sampled point counts per batch with shape (batch_size,).
            Format: [M1, M2, ...].
        features : torch.Tensor or None, optional
            Input features with shape (N1+N2+..., C). Default is None.
        empty_voxel_set_zeros : bool, optional
            If True, sets empty voxel features to zero. Default is True.

        Returns
        -------
        new_xyz : torch.Tensor
            Output point coordinates with shape (M1+M2+..., 3).
        new_features : torch.Tensor
            Aggregated multi-scale features with shape (M1+M2+..., C_out).
            C_out = sum(mlps[k][-1] for all scales k).
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == "max_pool":
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == "avg_pool":
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):
    """
    Feature propagation module for upsampling point features.

    Propagates features from coarse to fine levels using inverse distance
    weighted interpolation followed by MLP processing.

    Parameters
    ----------
    mlp : List[int]
        MLP specification defining layer dimensions [C_in, C_1, ..., C_out].

    Attributes
    ----------
    mlp : nn.Sequential
        MLP network with Conv2d, BatchNorm2d, and ReLU layers.
    """

    def __init__(self, *, mlp: List[int]):
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
        self,
        unknown: torch.Tensor,
        unknown_batch_cnt: torch.Tensor,
        known: torch.Tensor,
        known_batch_cnt: torch.Tensor,
        unknown_feats: Optional[torch.Tensor] = None,
        known_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Propagate features from known to unknown points via interpolation.

        Parameters
        ----------
        unknown : torch.Tensor
            Target point coordinates with shape (N1+N2+..., 3).
        unknown_batch_cnt : torch.Tensor
            Target point counts per batch with shape (batch_size,).
            Format: [N1, N2, ...].
        known : torch.Tensor
            Source point coordinates with shape (M1+M2+..., 3).
        known_batch_cnt : torch.Tensor
            Source point counts per batch with shape (batch_size,).
            Format: [M1, M2, ...].
        unknown_feats : torch.Tensor or None, optional
            Target point features with shape (N1+N2+..., C1). Default is None.
        known_feats : torch.Tensor or None, optional
            Source point features with shape (M1+M2+..., C2). Default is None.

        Returns
        -------
        new_features : torch.Tensor
            Propagated features with shape (N1+N2+..., C_out).
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features
