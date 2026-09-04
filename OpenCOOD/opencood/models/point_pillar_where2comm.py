import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.communication_adapters.where2comm import Where2CommCommunicationAdapter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarWhere2comm(nn.Module):
    communication_adapter_class = Where2CommCommunicationAdapter

    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()
        self.max_cav = args["max_cav"]
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        # Used to down-sample the feature map for efficient computation
        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        else:
            self.shrink_flag = False

        if args["compression"]:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
        else:
            self.compression = False

        self.fusion_net = Where2comm(args["where2comm_fusion"])
        self.multi_scale = args["where2comm_fusion"]["multi_scale"]

        self.cls_head = nn.Conv2d(args["head_dim"], args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(args["head_dim"], 7 * args["anchor_number"], kernel_size=1)

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
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

    def forward(self, data_dict):
        if "intermediate_features" in data_dict:
            return self.fuse_agents(data_dict)

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

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict["spatial_features_2d"]
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = self.fusion_net(
                batch_dict["spatial_features"], psm_single, record_len, pairwise_t_matrix, self.backbone
            )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d, psm_single, record_len, pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {"psm": psm, "rm": rm, "com": communication_rates}
        return output_dict

    def encode_agent(self, data_dict, *, apply_communication_mask):
        """
        Encode one agent up to the Where2Comm transmission boundary.

        Parameters
        ----------
        data_dict : dict
            Sender-local preprocessed LiDAR input.
        apply_communication_mask : bool
            Whether to select only features intended for transmission.

        Returns
        -------
        dict
            Learned feature map and selected-location ratio.
        """
        processed_lidar = data_dict["processed_lidar"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": torch.ones(1, dtype=torch.long, device=processed_lidar["voxel_features"].device),
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        confidence_features = batch_dict["spatial_features_2d"]
        if self.shrink_flag:
            confidence_features = self.shrink_conv(confidence_features)
        confidence_map = self.cls_head(confidence_features)

        if self.multi_scale:
            spatial_features = self.backbone.blocks[0](batch_dict["spatial_features"])
        else:
            spatial_features = confidence_features
            if self.compression:
                spatial_features = self.naive_compressor.encode(spatial_features)

        communication_rate = torch.ones((), device=spatial_features.device)
        if not self.fusion_net.fully:
            communication_mask, communication_rate = self.fusion_net.naive_communication.build_mask(
                confidence_map,
                force_first=False,
            )
            if apply_communication_mask:
                if spatial_features.shape[-2:] != communication_mask.shape[-2:]:
                    communication_mask = F.interpolate(
                        communication_mask,
                        size=spatial_features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                spatial_features = spatial_features * communication_mask

        return {
            "spatial_features": spatial_features,
            "communication_rate": communication_rate,
        }

    def fuse_agents(self, data_dict):
        """
        Fuse successfully received learned features and run detection heads.

        Parameters
        ----------
        data_dict : dict
            Receiver input containing encoded features and ``record_len``.

        Returns
        -------
        dict
            Fused classification, regression, and communication outputs.
        """
        intermediate_features = data_dict["intermediate_features"]
        spatial_features = intermediate_features["spatial_features"]
        record_len = data_dict["record_len"]

        if self.multi_scale:
            batch_node_features = self.fusion_net.regroup(spatial_features, record_len)
            fused_feature = torch.stack([self.fusion_net.fuse_modules[0](features) for features in batch_node_features])
            upsampled_features = [self.backbone.deblocks[0](fused_feature) if len(self.backbone.deblocks) > 0 else fused_feature]

            for level in range(1, self.fusion_net.num_levels):
                spatial_features = self.backbone.blocks[level](spatial_features)
                batch_node_features = self.fusion_net.regroup(spatial_features, record_len)
                fused_feature = torch.stack([self.fusion_net.fuse_modules[level](features) for features in batch_node_features])
                if len(self.backbone.deblocks) > 0:
                    upsampled_features.append(self.backbone.deblocks[level](fused_feature))
                else:
                    upsampled_features.append(fused_feature)

            fused_feature = torch.cat(upsampled_features, dim=1)
            if len(self.backbone.deblocks) > self.fusion_net.num_levels:
                fused_feature = self.backbone.deblocks[-1](fused_feature)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            if self.compression:
                spatial_features = self.naive_compressor.decode(spatial_features)
            batch_node_features = self.fusion_net.regroup(spatial_features, record_len)
            fused_feature = torch.stack([self.fusion_net.fuse_modules(features) for features in batch_node_features])

        communication_rates = intermediate_features["communication_rate"]
        return {
            "psm": self.cls_head(fused_feature),
            "rm": self.reg_head(fused_feature),
            "com": communication_rates.mean(),
        }
