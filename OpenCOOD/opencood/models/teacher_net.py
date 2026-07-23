import torch.nn as nn
from opencood.models.sub_modules.disconet_backbone import DiscoNetBackbone


class TeacherNet(nn.Module):
    """
    Teacher model for DiscoNet.
    """

    def __init__(self, args):
        super(TeacherNet, self).__init__()

        self.backbone = DiscoNetBackbone(args)

        self.category_num = args["category_num"]
        self.anchor_num = len(args["anchor_size"])
        self.cls_head = nn.Conv2d(32, self.category_num * self.anchor_num, kernel_size=1)
        self.reg_head = nn.Conv2d(32, 6 * self.anchor_num, kernel_size=1)

    def forward(self, data_dict):
        bevs = data_dict["bev_seq"]
        encoded_layers = self.backbone.encode(bevs)
        decoded_output = self.backbone.decode(encoded_layers, bevs.shape[0], kd_flag=True)
        main_feature = decoded_output[0]
        cls_preds = self.cls_head(main_feature).permute(0, 2, 3, 1).contiguous()
        reg_preds = self.reg_head(main_feature).permute(0, 2, 3, 1).contiguous()

        return {
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "fused_layer": encoded_layers[3],
            "x_7": decoded_output[1],
            "x_6": decoded_output[2],
            "x_5": decoded_output[3],
        }
