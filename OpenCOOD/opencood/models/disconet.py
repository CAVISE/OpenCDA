import torch.nn as nn

from opencood.models.sub_modules.disconet_backbone import DiscoNetBackbone
from opencood.models.fuse_modules.disconet_fuse import DiscoNetFusion


class DiscoNet(nn.Module):
    """
    Main model class for DiscoNet in OpenCOOD.
    """

    def __init__(self, args):
        super(DiscoNet, self).__init__()

        self.backbone = DiscoNetBackbone(args)

        self.fusion_net = DiscoNetFusion(args)

        self.category_num = args["category_num"]
        self.anchor_num = len(args["anchor_size"])

        self.cls_head = nn.Conv2d(32, self.category_num * self.anchor_num, kernel_size=1)
        self.reg_head = nn.Conv2d(32, 6 * self.anchor_num, kernel_size=1)

    def forward(self, data_dict):
        """
        data_dict contains:
            - 'bev_seq': (B, L, Z, H, W) voxel tensor
            - 'pairwise_t_matrix': (B, L, L, 4, 4) transformation matrices
            - 'record_len': (B) number of agents in each scene
        """
        bevs = data_dict["bev_seq"]  # (B*L, Seq, Z, H, W)
        record_len = data_dict["record_len"]  # (B)
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]  # (B, L, L, 4, 4)

        kd_flag = data_dict.get("kd_flag", False)

        total_agent_samples = bevs.shape[0]

        encoded_layers = self.backbone.encode(bevs)

        feature_to_fuse = encoded_layers[3]

        fused_feature = self.fusion_net(feature_to_fuse, record_len, pairwise_t_matrix)

        encoded_layers[3] = fused_feature

        decoded_output = self.backbone.decode(encoded_layers, total_agent_samples, kd_flag=kd_flag)

        main_feature = decoded_output[0]

        cls_preds = self.cls_head(main_feature)
        reg_preds = self.reg_head(main_feature)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous()

        output_dict = {"cls_preds": cls_preds, "reg_preds": reg_preds}

        if kd_flag:
            output_dict.update({"fused_layer": fused_feature, "x_7": decoded_output[1], "x_6": decoded_output[2], "x_5": decoded_output[3]})

        return output_dict
