"""
PointNet++ set abstraction and feature propagation modules.

This module provides PyTorch implementations of PointNet++ layers including
multi-scale set abstraction (MSG), single-scale set abstraction (SA), and
feature propagation (FP) for hierarchical point cloud processing.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.pcdet_utils.pointnet2.pointnet2_batch import pointnet2_utils


class _PointnetSAModuleBase(nn.Module):
    """
    Base class for PointNet++ set abstraction modules.
    
    Provides common forward pass logic for set abstraction with
    multi-scale grouping and pooling.

    Attributes
    ----------
    npoint : int or None
        Number of points to sample. None means use all points.
    groupers : nn.ModuleList or None
        List of grouping modules for each scale.
    mlps : nn.ModuleList or None
        List of MLP networks for each scale.
    pool_method : str
        Pooling method: 'max_pool' or 'avg_pool'.
    """

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = "max_pool"

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Set abstraction with multi-scale feature extraction.

        Parameters
        ----------
        xyz : torch.Tensor
            Input point coordinates with shape (B, N, 3).
        features : torch.Tensor or None, optional
            Input features with shape (B, N, C). Default is None.
        new_xyz : torch.Tensor or None, optional
            Pre-sampled point coordinates with shape (B, npoint, 3).
            If None, performs furthest point sampling. Default is None.

        Returns
        -------
        new_xyz : torch.Tensor
            Sampled point coordinates with shape (B, npoint, 3).
        new_features : torch.Tensor
            Aggregated features with shape (B, npoint, C_out).
            C_out = sum(mlps[k][-1] for all scales k).
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = (
                pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous()
                if self.npoint is not None
                else None
            )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == "max_pool":
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == "avg_pool":
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """
    PointNet++ set abstraction layer with multi-scale grouping (MSG).
    
    Extracts features at multiple scales using different ball query radii
    and sample counts, then concatenates results.

    Parameters
    ----------
    npoint : int
        Number of points to sample via furthest point sampling.
    radii : List[float]
        List of ball query radii for each scale.
    nsamples : List[int]
        List of maximum sample counts per ball for each scale.
    mlps : List[List[int]]
        List of MLP specifications for each scale.
        Each inner list defines layer dimensions [C_in, C_1, ..., C_out].
    bn : bool, optional
        Whether to use batch normalization. Default is True.
    use_xyz : bool, optional
        If True, concatenates xyz coordinates to features. Default is True.
    pool_method : str, optional
        Pooling method: 'max_pool' or 'avg_pool'. Default is 'max_pool'.

    Attributes
    ----------
    npoint : int
        Stored number of points to sample.
    groupers : nn.ModuleList
        List of QueryAndGroup modules for each scale.
    mlps : nn.ModuleList
        List of MLP networks for each scale.
    pool_method : str
        Stored pooling method.
    """

    def __init__(
        self,
        *,
        npoint: int,
        radii: List[float],
        nsamples: List[int],
        mlps: List[List[int]],
        bn: bool = True,
        use_xyz: bool = True,
        pool_method="max_pool",
    ):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """
    PointNet++ set abstraction layer with single-scale grouping.
    
    Simplified version of MSG with a single radius and sample count.
    Inherits from PointnetSAModuleMSG with single-element lists.

    Parameters
    ----------
    mlp : List[int]
        MLP specification defining layer dimensions [C_in, C_1, ..., C_out].
    npoint : int or None, optional
        Number of points to sample. None means use all points. Default is None.
    radius : float or None, optional
        Ball query radius. Default is None.
    nsample : int or None, optional
        Maximum number of samples per ball. Default is None.
    bn : bool, optional
        Whether to use batch normalization. Default is True.
    use_xyz : bool, optional
        If True, concatenates xyz coordinates to features. Default is True.
    pool_method : str, optional
        Pooling method: 'max_pool' or 'avg_pool'. Default is 'max_pool'.

    Attributes
    ----------
    Inherits all attributes from PointnetSAModuleMSG.
    """

    def __init__(
        self,
        *,
        mlp: List[int],
        npoint: int = None,
        radius: float = None,
        nsample: int = None,
        bn: bool = True,
        use_xyz: bool = True,
        pool_method="max_pool",
    ):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, pool_method=pool_method)


class PointnetFPModule(nn.Module):
    """
    Feature propagation module for upsampling point features.
    
    Propagates features from coarse to fine levels using inverse distance
    weighted interpolation followed by MLP processing.

    Parameters
    ----------
    mlp : List[int]
        MLP specification defining layer dimensions [C_in, C_1, ..., C_out].
    bn : bool, optional
        Whether to use batch normalization. Default is True.

    Attributes
    ----------
    mlp : nn.Sequential
        MLP network with Conv2d, BatchNorm2d, and ReLU layers.
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        """
        Propagate features from known to unknown points via interpolation.

        Parameters
        ----------
        unknown : torch.Tensor
            Target point coordinates with shape (B, n, 3).
        known : torch.Tensor
            Source point coordinates with shape (B, m, 3).
        unknow_feats : torch.Tensor or None
            Target point features with shape (B, C1, n).
        known_feats : torch.Tensor
            Source point features with shape (B, C2, m).

        Returns
        -------
        new_features : torch.Tensor
            Propagated features with shape (B, mlp[-1], n).
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
