"""
Downsampling Convolution Blocks.

This module implements double convolution blocks and sequential downsampling
using configurable kernel sizes, strides, and padding.
"""

from typing import Any, Dict, Union, Tuple
import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    """
    Double convolution block with ReLU activations.

    This block applies two consecutive 2D convolutions with ReLU activations.
    The first convolution is configurable, while the second uses a fixed 3x3 kernel.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Kernel size for the first convolution.
    stride : int or tuple of int, optional
        Stride for the first convolution. Default is 1.
    padding : int or tuple of int, optional
        Padding for the first convolution. Default is 0.

    Attributes
    ----------
    double_conv : nn.Sequential
        Sequential container with two Conv2d-ReLU pairs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through double convolution block.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, C_in, H, W).

        Returns
        -------
        Tensor
            Output features with shape (B, C_out, H', W').
        """
        return self.double_conv(x)


class DownsampleConv(nn.Module):
    """
    A sequence of downsampling convolution blocks.

    Parameters
    ----------
    config : Dict[str, List[Union[int, Tuple[int, int]]]]
        Configuration dictionary containing:

        - input_dim : int
            Number of input channels.
        - kernal_size : list
            List of kernel sizes for each block.
        - dim : list
            List of output dimensions for each block.
        - stride : list
            List of stride values for each block.
        - padding : list
            List of padding values for each block.

    Attributes
    ----------
    layers : nn.ModuleList
        List of DoubleConv blocks for sequential processing.
    """

    def __init__(self, config: Dict[str, Any]):
        super(DownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])
        input_dim = config["input_dim"]

        for ksize, dim, stride, padding in zip(config["kernal_size"], config["dim"], config["stride"], config["padding"]):
            self.layers.append(DoubleConv(input_dim, dim, kernel_size=ksize, stride=stride, padding=padding))
            input_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through all downsampling blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after all downsampling operations.
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
