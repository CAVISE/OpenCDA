"""
ResNet-based BEV Backbone with Multi-scale Feature Decoding.

This module implements a ResNet-based backbone for Bird's Eye View (BEV)
feature extraction with multi-scale upsampling and feature fusion.
"""

import numpy as np
import torch
import torch.nn as nn
from opencood.models.sub_modules.resblock import ResNetLayers

from typing import Dict, Any, Tuple


class ResBEVBackbone(nn.Module):
    """
    ResNet-based BEV backbone with multi-scale feature extraction.

    This backbone processes BEV features through multiple ResNet layers
    and upsamples them to a common resolution for feature fusion.

    Parameters
    ----------
    model_cfg : dict
        Model configuration dictionary containing:
        - 'layer_nums': Number of blocks in each ResNet layer.
        - 'layer_strides': Stride for each ResNet layer.
        - 'num_filters': Output channels for each ResNet layer.
        - 'upsample_strides': Upsampling strides for each layer (optional).
        - 'num_upsample_filter': Output channels after upsampling (optional).
    input_channels : int, optional
        Number of input feature channels. Default is 64.

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    resnet : ResNetLayers
        ResNet layers for feature extraction.
    num_levels : int
        Number of ResNet layers.
    deblocks : nn.ModuleList
        List of upsampling/downsampling blocks for each layer.
    num_bev_features : int
        Total number of output BEV feature channels.
    """

    def __init__(self, model_cfg: Dict[str, Any], input_channels: int = 64):
        super().__init__()
        self.model_cfg = model_cfg

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

        self.resnet = ResNetLayers(layer_nums, layer_strides, num_filters, inplanes=input_channels)

        num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
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
                    stride = np.round(1 / stride).astype(np.int)
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

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for single-agent detection (late fusion).

        For intermediate fusion, use get_multiscale_feature and
        decode_multiscale_feature instead.

        Parameters
        ----------
        data_dict : dict of str to Any
            Data dictionary containing:
            - 'spatial_features': BEV features with shape (B, C, H, W).

        Returns
        -------
        dict of str to Any
            Updated data dictionary with:
            - 'spatial_features_2d': Fused BEV features with shape (B, C_out, H', W').
        """
        spatial_features = data_dict["spatial_features"]

        x = self.resnet(spatial_features)  # tuple of features
        ups = []

        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict["spatial_features_2d"] = x
        return data_dict

    def decode_multiscale_feature(self, x):
        """
        Decode and fuse multi-scale features after intermediate fusion.

        Parameters
        ----------
        x : tuple of Tensor
            Multi-scale features from each ResNet layer.

        Returns
        -------
        Tensor
            Fused BEV features with shape (B, C_out, H', W').
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x

    def get_layer_i_feature(self, spatial_features: torch.Tensor, layer_i: int) -> torch.Tensor:
        """
        Extract features from a specific ResNet layer.

        Parameters
        ----------
        spatial_features : Tensor
            Input features with shape (B, C, H, W).
        layer_i : int
            Layer index to extract features from.

        Returns
        -------
        Tensor
            Features from the specified layer with shape (B, C_i, H_i, W_i).
        """
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)  # tuple of features
