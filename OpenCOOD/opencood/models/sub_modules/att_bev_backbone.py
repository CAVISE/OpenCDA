"""
Attention-based BEV Backbone with Multi-scale Feature Fusion.

This module implements a BEV backbone that uses attention mechanisms to fuse
features from multiple agents at different scales, with optional compression
for efficient communication.
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Any, Dict
from torch import Tensor

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


class AttBEVBackbone(nn.Module):
    """
    Attention-based BEV Backbone with multi-scale feature extraction and fusion.

    Parameters
    ----------
    model_cfg : dict
        Dictionary containing model configuration:

        - layer_nums : list of int
            Number of layers in each block.
        - layer_strides : list of int
            Stride for each block.
        - num_filters : list of int
            Number of filters for each block.
        - upsample_strides : list of int
            Upsampling factors.
        - num_upsample_filter : list of int
            Number of filters for upsampling.
        - compression : int, optional
            Compression ratio for autoencoder.
    input_channels : int
        Number of input channels.

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    compress : bool
        Whether feature compression is enabled.
    compress_layer : int, optional
        Number of compression layers (if compression is enabled).
    blocks : nn.ModuleList
        List of convolutional blocks for feature extraction.
    fuse_modules : nn.ModuleList
        List of attention fusion modules for multi-agent fusion.
    deblocks : nn.ModuleList
        List of upsampling blocks.
    compression_modules : nn.ModuleList, optional
        List of autoencoder compression modules (if compression is enabled).
    num_bev_features : int
        Total number of output BEV feature channels.
    """

    def __init__(self, model_cfg: Dict[str, Any], input_channels: int):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if "compression" in model_cfg and model_cfg["compression"] > 0:
            self.compress = True
            self.compress_layer = model_cfg["compression"]

        if "layer_nums" in self.model_cfg:
            assert len(self.model_cfg["layer_nums"]) == len(self.model_cfg["layer_strides"]) == len(self.model_cfg["num_filters"])

            layer_nums = self.model_cfg["layer_nums"]
            layer_strides = self.model_cfg["layer_strides"]
            num_filters = self.model_cfg["num_filters"]
        else:
            layer_nums = layer_strides = num_filters = []

        if "upsample_strides" in self.model_cfg:
            assert len(self.model_cfg["upsample_strides"]) == len(self.model_cfg["num_upsample_filter"])

            num_upsample_filters = self.model_cfg["num_upsample_filter"]
            upsample_strides = self.model_cfg["upsample_strides"]

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx], self.compress_layer - idx))

            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = int(np.round(1 / stride))
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for the BEV backbone.

        Parameters
        ----------
        data_dict : dict of str to Tensor
            Data dictionary containing:
            - 'spatial_features': Input BEV features with shape (B, C, H, W).
            - 'record_len': Number of agents per batch for attention fusion.

        Returns
        -------
        dict of str to Tensor
            Updated data dictionary with:
            - 'spatial_features_2d': Fused multi-scale features with shape (B, C_out, H', W').
            - 'spatial_features_Nx': Intermediate features at each scale before fusion
        """
        spatial_features = data_dict["spatial_features"]
        record_len = data_dict["record_len"]

        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)
            x_fuse = self.fuse_modules[i](x, record_len)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict["spatial_features_%dx" % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict["spatial_features_2d"] = x
        return data_dict
