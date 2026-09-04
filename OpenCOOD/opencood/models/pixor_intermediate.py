import math

import torch.nn as nn

from opencood.models.communication_adapters.intermediate import MultiScaleFeatureCommunicationAdapter
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.pixor import Bottleneck, BackBone, Header


class BackBoneIntermediate(BackBone):
    def __init__(self, block, num_block, geom, use_bn=True):
        super(BackBoneIntermediate, self).__init__(block, num_block, geom, use_bn)

        self.fusion_net3 = AttFusion(192)
        self.fusion_net4 = AttFusion(256)
        self.fusion_net5 = AttFusion(384)

    def forward(self, x, record_len):
        return self.fuse_agents(self.encode(x), record_len)

    def fuse_agents(self, feature_maps, record_len):
        """
        Fuse encoded PIXOR scales and run the feature pyramid decoder.

        Parameters
        ----------
        feature_maps : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Per-agent ``c3``, ``c4``, and ``c5`` feature maps.
        record_len : torch.Tensor
            Number of available agents in each sample.

        Returns
        -------
        torch.Tensor
            Decoded fused PIXOR feature map.
        """
        c3, c4, c5 = feature_maps
        c5 = self.fusion_net5(c5, record_len)
        c4 = self.fusion_net4(c4, record_len)
        c3 = self.fusion_net3(c3, record_len)
        return self.decode(c3, c4, c5)


class PIXORIntermediate(nn.Module):
    """
    The Pixor backbone. The input of PIXOR nn module is a tensor of
    [batch_size, height, weight, channel], The output of PIXOR nn module
    is also a tensor of [batch_size, height/4, weight/4, channel].  Note that
     we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions

    Parameters
    ----------
    args : dict
        The arguments of the model.

    Attributes
    ----------
    backbone : opencood.object
        The backbone used to extract features.
    header : opencood.object
        Header used to predict the classification and coordinates.
    """

    communication_adapter_class = MultiScaleFeatureCommunicationAdapter

    def __init__(self, args):
        super(PIXORIntermediate, self).__init__()
        geom = args["geometry_param"]
        use_bn = args["use_bn"]
        self.backbone = BackBoneIntermediate(Bottleneck, [3, 6, 6, 3], geom, use_bn)
        self.header = Header(use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.header.clshead.weight.data.fill_(-math.log((1.0 - prior) / prior))
        self.header.clshead.bias.data.fill_(0)
        self.header.reghead.weight.data.fill_(0)
        self.header.reghead.bias.data.fill_(0)

    def forward(self, data_dict):
        if "intermediate_features" in data_dict:
            return self.fuse_agents(data_dict)

        bev_input = data_dict["processed_lidar"]["bev_input"]
        record_len = data_dict["record_len"]

        features = self.backbone(bev_input, record_len)
        # cls -- (N, 1, W/4, L/4)
        # reg -- (N, 6, W/4, L/4)
        cls, reg = self.header(features)

        output_dict = {"cls": cls, "reg": reg}

        return output_dict

    def encode_agent(self, data_dict):
        """
        Encode one agent into PIXOR's three private backbone scales.

        Parameters
        ----------
        data_dict : dict
            Sender-local preprocessed BEV input.

        Returns
        -------
        dict
            Ordered ``c3``, ``c4``, and ``c5`` feature maps.
        """
        return {
            "feature_maps": self.backbone.encode(data_dict["processed_lidar"]["bev_input"]),
        }

    def fuse_agents(self, data_dict):
        """
        Fuse received PIXOR features and run its detection header.

        Parameters
        ----------
        data_dict : dict
            Receiver input containing learned features and ``record_len``.

        Returns
        -------
        dict
            PIXOR classification and regression maps.
        """
        intermediate_features = data_dict["intermediate_features"]
        feature_maps = tuple(intermediate_features[f"feature_{index}"] for index in range(3))
        features = self.backbone.fuse_agents(feature_maps, data_dict["record_len"])
        cls, reg = self.header(features)
        return {"cls": cls, "reg": reg}
