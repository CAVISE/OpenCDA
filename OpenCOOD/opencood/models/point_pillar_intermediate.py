"""
PointPillar with intermediate attention-based fusion for 3D object detection.

This module implements PointPillar architecture with an attention-based BEV backbone
for intermediate feature extraction and multi-agent cooperative perception.
"""

import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone

from typing import Dict, Any

class PointPillarIntermediate(nn.Module):
    """
    PointPillar with intermediate attention-based feature extraction.

    This module implements a PointPillar-based architecture with an attention-based
    BEV backbone for feature extraction, followed by detection heads for classification
    and regression.

    Parameters
    ----------
    args : dict of str to Any
        Configuration dictionary containing:
        - 'pillar_vfe': Configuration for PillarVFE.
        - 'voxel_size': Voxel size [x, y, z].
        - 'lidar_range': LiDAR range [x_min, y_min, z_min, x_max, y_max, z_max].
        - 'point_pillar_scatter': Configuration for point pillar scatter.
        - 'base_bev_backbone': Configuration for attention-based BEV backbone.
        - 'anchor_number': Number of anchor boxes per position for classification.
        - 'anchor_num': Number of anchor boxes for regression head.

    Attributes
    ----------
    pillar_vfe : PillarVFE
        Pillar voxel feature encoder module.
    scatter : PointPillarScatter
        Scatter module to convert pillar features to pseudo-image.
    backbone : AttBEVBackbone
        Attention-based 2D BEV backbone for feature extraction.
    cls_head : nn.Conv2d
        Classification head for predicting object scores.
    reg_head : nn.Conv2d
        Regression head for predicting bounding box parameters.
    """

    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = AttBEVBackbone(args["base_bev_backbone"], 64)

        self.cls_head = nn.Conv2d(128 * 3, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass of the PointPillarIntermediate model.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing:
            - 'processed_lidar': Dictionary with 'voxel_features', 'voxel_coords',
              and 'voxel_num_points'.
            - 'record_len': Tensor indicating number of agents per batch sample.

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
        record_len = data_dict["record_len"]

        batch_dict = {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points, "record_len": record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
