import torch.nn as nn

from opencood.models.communication_adapters.intermediate import MultiScaleFeatureCommunicationAdapter
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class PointPillarIntermediate(nn.Module):
    communication_adapter_class = MultiScaleFeatureCommunicationAdapter

    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = AttBEVBackbone(args["base_bev_backbone"], 64)

        self.cls_head = nn.Conv2d(128 * 3, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_num"], kernel_size=1)

    def forward(self, data_dict):
        if "intermediate_features" in data_dict:
            return self.fuse_agents(data_dict)

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

    def encode_agent(self, data_dict):
        """
        Encode one agent into private multi-scale PointPillar features.

        Parameters
        ----------
        data_dict : dict
            Sender-local preprocessed LiDAR input.

        Returns
        -------
        dict
            Ordered feature maps at the model's attention boundaries.
        """
        processed_lidar = data_dict["processed_lidar"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        return {
            "feature_maps": self.backbone.encode_agent(batch_dict["spatial_features"]),
        }

    def fuse_agents(self, data_dict):
        """
        Fuse received feature scales and run PointPillar detection heads.

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
        feature_maps = tuple(intermediate_features[f"feature_{index}"] for index in range(len(self.backbone.fuse_modules)))
        fused_features = self.backbone.fuse_agents(feature_maps, data_dict["record_len"])
        return {
            "psm": self.cls_head(fused_features),
            "rm": self.reg_head(fused_features),
        }
