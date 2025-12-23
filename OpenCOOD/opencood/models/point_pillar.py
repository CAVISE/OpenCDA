"""
PointPillar model for efficient 3D object detection from point clouds.
This module implements the PointPillar architecture which processes point cloud data
through a pillar-based feature encoder, scatter operation, and BEV (Bird's Eye View)
backbone for efficient 3D object detection. 
"""
import torch.nn as nn
from typing import Dict, Any
import torch

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class PointPillar(nn.Module):
    """
    Standard PointPillar model for 3D object detection.
    This module implements the PointPillar architecture which processes point clouds
    through a pillar-based feature encoder, scatter operation, BEV backbone,
    and detection heads for classification and regression.
    """
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize the PointPillar model.
        Args:
            args: Configuration dictionary containing:
                - pillar_vfe: Configuration for PillarVFE
                - voxel_size: Voxel size [x, y, z]
                - lidar_range: LiDAR range [x_min, y_min, z_min, x_max, y_max, z_max]
                - point_pillar_scatter: Configuration for point pillar scatter
                - base_bev_backbone: Configuration for BaseBEVBackbone
                - anchor_number: Number of anchor boxes per position
                - anchor_num: Number of anchor boxes (used for regression head)
        """
        super(PointPillar, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        self.cls_head = nn.Conv2d(128 * 3, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]

        batch_dict = {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
