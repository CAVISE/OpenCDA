"""
Class used to downsample features by 3*3 conv
"""
from typing import Dict, List, Union, Optional, Tuple
import torch.nn as nn
from torch import Tensor


class DoubleConv(nn.Module):
    """
    Double convolution block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
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
    """
    def __init__(self, config: Dict[str, List[Union[int, Tuple[int, int]]]]):
        super(DownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])
        input_dim = config["input_dim"]

        for ksize, dim, stride, padding in zip(config["kernal_size"], config["dim"], config["stride"], config["padding"]):
            self.layers.append(DoubleConv(input_dim, dim, kernel_size=ksize, stride=stride, padding=padding))
            input_dim = dim

    def forward(self, x: Tensor) -> Tensor:
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
