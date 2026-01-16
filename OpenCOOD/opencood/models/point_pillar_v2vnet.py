"""
PointPillar with V2VNet for multi-agent collaborative 3D object detection.

This module implements PointPillar architecture integrated with V2VNet fusion
for multi-agent cooperative perception with efficient feature sharing.
"""

from typing import Dict, Any
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion


class PointPillarV2VNet(nn.Module):
    """
    PointPillar with V2VNet for multi-agent collaborative 3D object detection.

    This model combines PointPillar architecture with V2VNet fusion mechanism
    to enable multi-agent cooperative perception through efficient feature aggregation.

    Parameters
    ----------
    args : Dict[str, Any]
        Configuration dictionary containing:
            - max_cav: Maximum number of connected automated vehicles
            - pillar_vfe: Configuration for PillarVFE
            - voxel_size: Voxel size [x, y, z]
            - lidar_range: LiDAR range [x_min, y_min, z_min, x_max, y_max, z_max]
            - point_pillar_scatter: Configuration for point pillar scatter
            - base_bev_backbone: Configuration for BaseBEVBackbone
            - shrink_header: Optional configuration for feature downsampling
            - compression: Compression dimension (0 for no compression, >0 for compression)
            - v2vfusion: Configuration for V2VNet fusion
            - anchor_number: Number of anchor boxes per position
            - backbone_fix: Whether to fix backbone parameters during training
    
    Attributes
    ----------
    max_cav : int
        Maximum number of connected automated vehicles.
    pillar_vfe : PillarVFE
        Pillar voxel feature encoder module.
    scatter : PointPillarScatter
        Scatter module to convert pillar features to pseudo-image.
    backbone : BaseBEVBackbone
        2D backbone network for BEV feature extraction.
    shrink_flag : bool
        Flag indicating whether feature downsampling is enabled.
    shrink_conv : DownsampleConv, optional
        Downsampling convolution module if shrink_flag is True.
    compression : bool
        Flag indicating whether feature compression is enabled.
    naive_compressor : NaiveCompressor, optional
        Feature compression module if compression is enabled.
    fusion_net : V2VNetFusion
        V2VNet fusion module for multi-agent feature fusion.
    cls_head : nn.Conv2d
        Classification head for predicting object scores.
    reg_head : nn.Conv2d
        Regression head for predicting bounding box parameters.
    """

    def __init__(self, args: Dict[str, Any]):
        super(PointPillarV2VNet, self).__init__()

        self.max_cav = args["max_cav"]
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        self.compression = False

        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.fusion_net = V2VNetFusion(args["v2vfusion"])

        self.cls_head = nn.Conv2d(128 * 2, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args["anchor_number"], kernel_size=1)
        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self) -> None:
        """
        Fix the parameters of backbone during fine-tuning on time delay.

        This method freezes gradients for all backbone components including
        pillar VFE, scatter, backbone, compressor, shrink conv, and detection heads.
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the PointPillarV2VNet model.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing:
            - 'processed_lidar': Dictionary with 'voxel_features', 'voxel_coords',
              and 'voxel_num_points'.
            - 'record_len': Tensor indicating number of agents per batch sample.
            - 'pairwise_t_matrix': Pairwise transformation matrices between agents.

        Returns
        -------
        dict of str to torch.Tensor
            Output dictionary with keys:
            - 'psm': Probability score map with shape (batch_size, anchor_number, H, W).
            - 'rm': Regression map with shape (batch_size, 7*anchor_number, H, W).
        """
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]

        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

        batch_dict = {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points, "record_len": record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        fused_feature = self.fusion_net(spatial_features_2d, record_len, pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
