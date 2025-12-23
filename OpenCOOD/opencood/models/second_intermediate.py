import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from typing import Dict, Any
import torch

class SecondIntermediate(nn.Module):
    """
    SECOND with intermediate feature extraction for 3D object detection.
    This model implements the SECOND (Sparsely Embedded Convolutional Detection)
    architecture with an attention-based BEV backbone for feature extraction,
    followed by detection heads for classification and regression.
    """
    def __init__(self, args: Dict[str, Any]) -> None:
        """
        Initialize the SecondIntermediate model.
        Args:
            args: Configuration dictionary containing:
                - batch_size: Number of samples per batch
                - mean_vfe: Configuration for MeanVoxelFeatureExtractor
                - backbone_3d: Configuration for 3D sparse backbone
                - grid_size: Grid size for 3D voxelization [X, Y, Z]
                - height_compression: Configuration for height compression
                - base_bev_backbone: Configuration for 2D BEV backbone
                - anchor_number: Number of anchor boxes per position
                - anchor_num: Number of anchor boxes (used for regression head)
        """
        super(SecondIntermediate, self).__init__()

        self.batch_size = args["batch_size"]
        # mean_vfe
        self.mean_vfe = MeanVFE(args["mean_vfe"], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args["backbone_3d"], 4, args["grid_size"])
        # height compression
        self.height_compression = HeightCompression(args["height_compression"])
        # base ben backbone
        self.backbone_2d = AttBEVBackbone(args["base_bev_backbone"], 256)

        # head
        self.cls_head = nn.Conv2d(256 * 2, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(256 * 2, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the SecondIntermediate model.
        """
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "batch_size": torch.sum(record_len).cpu().numpy(),
            "record_len": record_len,
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
