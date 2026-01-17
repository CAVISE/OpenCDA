"""
SECOND (Sparsely Embedded Convolutional Detection) for 3D object detection.

This module implements the SECOND architecture which processes point cloud data
through a sparse 3D convolutional network, followed by a 2D BEV (Bird's Eye View)
backbone for efficient 3D object detection.
"""

from typing import Any

import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class Second(nn.Module):
    """
    SECOND (Sparsely Embedded Convolutional Detection) model for 3D object detection.

    This module implements the SECOND architecture which processes point clouds through:
    1. Voxel Feature Extractor (VFE)
    2. 3D Sparse Convolutional Backbone
    3. Height Compression to create BEV features
    4. 2D BEV Backbone for feature extraction
    5. Detection heads for classification and regression

    Parameters
    ----------
    args : dict of str to Any
        Configuration dictionary containing model hyperparameters including:
        - 'batch_size': Batch size for processing.
        - 'mean_vfe': Configuration for mean voxel feature encoder.
        - 'backbone_3d': Configuration for 3D sparse backbone.
        - 'grid_size': Voxel grid size.
        - 'height_compression': Configuration for height compression.
        - 'base_bev_backbone': Configuration for 2D BEV backbone.
        - 'anchor_number': Number of anchors for classification head.
        - 'anchor_num': Number of anchors for regression head.

    Attributes
    ----------
    batch_size : int
        Batch size for processing.
    mean_vfe : MeanVFE
        Mean voxel feature encoder module.
    backbone_3d : VoxelBackBone8x
        3D sparse convolutional backbone.
    height_compression : HeightCompression
        Height compression module to convert 3D features to BEV.
    backbone_2d : BaseBEVBackbone
        2D BEV backbone for feature extraction.
    cls_head : nn.Conv2d
        Classification head for predicting object scores.
    reg_head : nn.Conv2d
        Regression head for predicting bounding box parameters.
    """

    def __init__(self, args):
        super(Second, self).__init__()

        self.batch_size = args["batch_size"]
        # mean_vfe
        self.mean_vfe = MeanVFE(args["mean_vfe"], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args["backbone_3d"], 4, args["grid_size"])
        # height compression
        self.height_compression = HeightCompression(args["height_compression"])
        # base bev backbone
        self.backbone_2d = BaseBEVBackbone(args["base_bev_backbone"], 256)

        # head
        self.cls_head = nn.Conv2d(256 * 2, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(256 * 2, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the Second model.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing:
            - 'processed_lidar': Dictionary with 'voxel_features', 'voxel_coords',
              and 'voxel_num_points'.

        Returns
        -------
        dict of str to torch.Tensor
            Output dictionary with keys:
            - 'psm': Probability score map with shape (batch_size, anchor_number, H, W).
            - 'rm': Regression map with shape (batch_size, 7*anchor_num, H, W).
        """
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "batch_size": self.batch_size,
        }

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
