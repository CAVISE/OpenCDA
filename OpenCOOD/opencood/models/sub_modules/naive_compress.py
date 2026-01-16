"""
Naive Feature Compressor for Collaborative Perception.

This module implements a simple channel-wise compression and decompression
strategy for reducing communication bandwidth in multi-agent systems.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn


class NaiveCompressor(nn.Module):
    """
    Naive compression that only compress on the channel.

    This module compresses features by reducing the channel dimension using
    a simple encoder-decoder architecture with convolutional layers.

    Parameters
    ----------
    input_dim : int
        Number of input feature channels.
    compress_ratio : int
        Compression ratio for reducing channels (e.g., 2 reduces channels by half).

    Attributes
    ----------
    encoder : nn.Sequential
        Encoder module that compresses features by reducing channels.
    decoder : nn.Sequential
        Decoder module that reconstructs features to original channel dimension.
    """

    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // compress_raito, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim // compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim // compress_raito, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compress and decompress input features.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Reconstructed features with shape (B, C, H, W).
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x
