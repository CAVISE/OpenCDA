import torch
import torch.nn as nn

from opencood.models.communication_adapters.intermediate import MultiScaleFeatureCommunicationAdapter
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class SecondIntermediate(nn.Module):
    communication_adapter_class = MultiScaleFeatureCommunicationAdapter

    def __init__(self, args):
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

    def forward(self, data_dict):
        if "intermediate_features" in data_dict:
            return self.fuse_agents(data_dict)

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

    def encode_agent(self, data_dict):
        """
        Encode one agent into private multi-scale SECOND features.

        Parameters
        ----------
        data_dict : dict
            Sender-local preprocessed LiDAR input.

        Returns
        -------
        dict
            Ordered feature maps at the 2D attention boundaries.
        """
        processed_lidar = data_dict["processed_lidar"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "batch_size": data_dict.get("batch_size", 1),
        }
        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        return {
            "feature_maps": self.backbone_2d.encode_agent(batch_dict["spatial_features"]),
        }

    def fuse_agents(self, data_dict):
        """
        Fuse received SECOND feature scales and run detection heads.

        Parameters
        ----------
        data_dict : dict
            Receiver input containing learned features and ``record_len``.

        Returns
        -------
        dict
            Classification and regression maps.
        """
        intermediate_features = data_dict["intermediate_features"]
        feature_maps = tuple(intermediate_features[f"feature_{index}"] for index in range(len(self.backbone_2d.fuse_modules)))
        fused_features = self.backbone_2d.fuse_agents(
            feature_maps,
            data_dict["record_len"],
        )
        return {
            "psm": self.cls_head(fused_features),
            "rm": self.reg_head(fused_features),
        }
