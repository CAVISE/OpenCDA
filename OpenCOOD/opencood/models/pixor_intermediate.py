"""
PIXOR Intermediate model for multi-agent collaborative 3D object detection.

This module implements PIXOR with intermediate attention-based feature fusion
for multi-agent cooperative perception using BEV representations.
"""

import math

import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.pixor import Bottleneck, BackBone, Header
from typing import Dict, Any, List, Type


class BackBoneIntermediate(BackBone):
    """
    Feature Pyramid Network backbone with intermediate attention-based fusion.

    This backbone extends the standard PIXOR backbone by adding attention-based
    fusion modules at multiple scales for multi-agent feature aggregation.

    Parameters
    ----------
    block : type
        Residual block type (typically Bottleneck).
    num_block : list of int
        Number of blocks in each layer.
    geom : dict of str to Any
        Geometry parameters containing 'input_shape' and 'label_shape'.
    use_bn : bool, optional
        Whether to use batch normalization. Default is True.

    Attributes
    ----------
    fusion_net3 : AttFusion
        Attention fusion module for layer 3 features.
    fusion_net4 : AttFusion
        Attention fusion module for layer 4 features.
    fusion_net5 : AttFusion
        Attention fusion module for layer 5 features.
    """

    def __init__(
        self, 
        block: Type[nn.Module], 
        num_block: List[int], 
        geom: Dict[str, Any], 
        use_bn: bool = True
    ):
        super(BackBoneIntermediate, self).__init__(block, num_block, geom, use_bn)

        self.fusion_net3 = AttFusion(192)
        self.fusion_net4 = AttFusion(256)
        self.fusion_net5 = AttFusion(384)

    def forward(self, x: torch.Tensor, record_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone with intermediate fusion.

        Parameters
        ----------
        x : torch.Tensor
            Input BEV tensor with shape (N, C, H, W).
        record_len : torch.Tensor
            Tensor indicating number of agents per batch sample.

        Returns
        -------
        torch.Tensor
            Multi-scale fused features with shape (N, 96, H/4, W/4).
        """
        c3, c4, c5 = self.encode(x)

        c5 = self.fusion_net5(c5, record_len)
        c4 = self.fusion_net4(c4, record_len)
        c3 = self.fusion_net3(c3, record_len)

        p4 = self.decode(c3, c4, c5)
        return p4


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

    def __init__(self, args: Dict[str, Any]) -> None:
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

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PIXOR Intermediate model.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing:
            - 'processed_lidar': Dictionary with 'bev_input' BEV representation.
            - 'record_len': Tensor indicating number of agents per batch sample.

        Returns
        -------
        dict of str to Tensor
            Output dictionary with keys:
            - 'cls': Classification scores with shape (N, 1, H/4, W/4).
            - 'reg': Regression parameters with shape (N, 6, H/4, W/4).
        """
        bev_input = data_dict["processed_lidar"]["bev_input"]
        record_len = data_dict["record_len"]

        features = self.backbone(bev_input, record_len)
        # cls -- (N, 1, W/4, L/4)
        # reg -- (N, 6, W/4, L/4)
        cls, reg = self.header(features)

        output_dict = {"cls": cls, "reg": reg}

        return output_dict
