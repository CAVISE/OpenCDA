"""
Mean Voxel Feature Encoder (VFE).

This module implements a simple mean-pooling strategy for encoding point
features within voxels by averaging all points in each voxel.
"""

from typing import Dict
import torch
from torch import nn


class MeanVFE(nn.Module):
    """
    Mean-pooling Voxel Feature Encoder.

    This encoder aggregates point features within each voxel by computing
    the mean of all points, providing a simple and efficient feature
    representation for voxelized point clouds.

    Parameters
    ----------
    model_cfg : dict
        Model configuration dictionary.
    num_point_features : int
        Number of input point feature channels.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    num_point_features : int
        Number of point feature channels (also output dimension).
    """

    def __init__(self, model_cfg: Dict, num_point_features, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Encode voxel features by computing mean of points within each voxel.

        Parameters
        ----------
        batch_dict : dict of str to Any
            Batch dictionary containing:
            - 'voxel_features': Point features with shape (N_voxels, N_points, C).
            - 'voxel_num_points': Number of valid points per voxel with
              shape (N_voxels,).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict of str to Any
            Updated batch dictionary with:
            - 'voxel_features': Mean-pooled voxel features with shape (N_voxels, C).
        """
        voxel_features, voxel_num_points = batch_dict["voxel_features"], batch_dict["voxel_num_points"]
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict["voxel_features"] = points_mean.contiguous()

        return batch_dict
