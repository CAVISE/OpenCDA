"""
Autoencoder module for feature compression and decompression.

This module implements a simple convolutional autoencoder that can be used
for feature compression in multi-agent perception systems.
"""

import torch.nn as nn
from torch import Tensor

class AutoEncoder(nn.Module):
    """
    An autoencoder for feature compression.
    
    The autoencoder reduces the spatial dimensions of the input features
    while maintaining the number of channels. It consists of multiple
    encoding and decoding layers with skip connections.
    
    Parameters
    ----------
    feature_num : int
        Number of input/output channels.
    layer_num : int
        Number of encoding/decoding layers.

    Attributes
    ----------
    feature_num : int
        Number of input/output channels.
    feature_stride : int
        Channel reduction factor (always 2).
    encoder : nn.ModuleList
        List of encoding blocks that downsample spatially and reduce channels.
    decoder : nn.ModuleList
        List of decoding blocks that upsample spatially and increase channels.
    """

    def __init__(self, feature_num: int, layer_num: int):
        super().__init__()
        self.feature_num = feature_num
        self.feature_stride = 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(layer_num):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]

            cur_layers.extend(
                [
                    nn.Conv2d(feature_num, feature_num // self.feature_stride, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feature_num // self.feature_stride, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ]
            )

            self.encoder.append(nn.Sequential(*cur_layers))
            feature_num = feature_num // self.feature_stride

        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [
                nn.Sequential(
                    nn.ConvTranspose2d(feature_num // 2, feature_num, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            ]

            cur_layers.extend(
                [
                    nn.Sequential(
                        nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, bias=False, padding=1),
                        nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    )
                ]
            )
            self.decoder.append(nn.Sequential(*cur_layers))
            feature_num //= 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through autoencoder.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Reconstructed features with shape (B, C, H, W).

        Notes
        -----
        The spatial dimensions are progressively reduced by 2^layer_num during
        encoding and restored during decoding. The channel dimension follows
        the pattern: C -> C/2 -> C/4 -> ... -> C/(2^layer_num) -> ... -> C.
        """
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        for i in range(len(self.decoder) - 1, -1, -1):
            x = self.decoder[i](x)

        return x
