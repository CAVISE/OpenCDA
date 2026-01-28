"""
PIXOR model for BEV-based 3D object detection.

This module implements PIXOR, a bird's-eye-view based single-stage detector for
3D object detection using Feature Pyramid Networks and residual blocks.
"""

import math

import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Type, Union
import torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """
    Create a 3x3 convolution with padding.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    out_planes : int
        Number of output channels.
    stride : int, optional
        Convolution stride. Default is 1.
    bias : bool, optional
        Whether to include bias term. Default is False.

    Returns
    -------
    nn.Conv2d
        2D convolution layer with kernel size 3 and padding 1.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet architecture.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    planes : int
        Number of output channels.
    stride : int, optional
        Convolution stride. Default is 1.
    downsample : nn.Module, optional
        Downsampling layer for residual connection. Default is None.

    Attributes
    ----------
    expansion : int
        Channel expansion factor (always 1 for BasicBlock).
    conv1 : nn.Conv2d
        First 3x3 convolution layer.
    bn1 : nn.BatchNorm2d
        First batch normalization layer.
    relu : nn.ReLU
        ReLU activation function.
    conv2 : nn.Conv2d
        Second 3x3 convolution layer.
    bn2 : nn.BatchNorm2d
        Second batch normalization layer.
    downsample : nn.Module or None
        Downsampling layer for residual connection.
    stride : int
        Convolution stride.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.torch.Tensor) -> torch.torch.Tensor:
        """
        Forward pass of BasicBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input torch.Tensor with shape (N, in_planes, H, W).

        Returns
        -------
        torch.Tensor
            Output torch.Tensor with shape (N, planes, H/stride, W/stride).
        """
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet architecture.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    planes : int
        Number of intermediate channels.
    stride : int, optional
        Convolution stride. Default is 1.
    downsample : nn.Module, optional
        Downsampling layer for residual connection. Default is None.
    use_bn : bool, optional
        Whether to use batch normalization. Default is True.

    Attributes
    ----------
    expansion : int
        Channel expansion factor (always 4 for Bottleneck).
    use_bn : bool
        Flag indicating whether batch normalization is used.
    conv1 : nn.Conv2d
        First 1x1 convolution layer.
    bn1 : nn.BatchNorm2d
        First batch normalization layer.
    conv2 : nn.Conv2d
        Second 3x3 convolution layer.
    bn2 : nn.BatchNorm2d
        Second batch normalization layer.
    conv3 : nn.Conv2d
        Third 1x1 convolution layer.
    bn3 : nn.BatchNorm2d
        Third batch normalization layer.
    downsample : nn.Module or None
        Downsampling layer for residual connection.
    relu : nn.ReLU
        ReLU activation function.
    """

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, use_bn: bool = True):
        super(Bottleneck, self).__init__()
        bias = not use_bn
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of residual block.
        Parameters
        ----------
        x : torch.torch.Tensor
            Shape (N, C, W, L).

        Returns
        -------
        out : torch.torch.Tensor
            Shape (N, self.expansion*planes, W/stride, L/stride).
        """
        residual = x
        # (N, planes, W, L)
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        # (N, planes, W/stride, L/stride)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)
        # (N, self.expansion*planes, W/stride, L/stride)
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            # (N, self.expansion*planes, W/2, L/2)
            residual = self.downsample(x)
        out = self.relu(residual + out)
        return out


class BackBone(nn.Module):
    """
    Feature Pyramid Network backbone for PIXOR.

    This backbone extracts multi-scale features using residual blocks and
    combines them using a top-down pathway with lateral connections.

    Parameters
    ----------
    block : type
        Residual block type (BasicBlock or Bottleneck).
    num_block : list of int
        Number of blocks in each layer.
    geom : dict of str to Any
        Geometry parameters containing 'input_shape' and 'label_shape'.
    use_bn : bool, optional
        Whether to use batch normalization. Default is True.

    Attributes
    ----------
    use_bn : bool
        Flag indicating whether batch normalization is used.
    conv1 : nn.Conv2d
        First 3x3 convolution layer.
    conv2 : nn.Conv2d
        Second 3x3 convolution layer.
    bn1 : nn.BatchNorm2d
        First batch normalization layer.
    bn2 : nn.BatchNorm2d
        Second batch normalization layer.
    relu : nn.ReLU
        ReLU activation function.
    in_planes : int
        Current number of input channels for layer construction.
    block2 : nn.Sequential
        Second residual block layer.
    block3 : nn.Sequential
        Third residual block layer.
    block4 : nn.Sequential
        Fourth residual block layer.
    block5 : nn.Sequential
        Fifth residual block layer.
    latlayer1 : nn.Conv2d
        Lateral connection for highest resolution features.
    latlayer2 : nn.Conv2d
        Lateral connection for middle resolution features.
    latlayer3 : nn.Conv2d
        Lateral connection for lower resolution features.
    deconv1 : nn.ConvTranspose2d
        First deconvolution layer for upsampling.
    deconv2 : nn.ConvTranspose2d
        Second deconvolution layer for upsampling.
    """

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], num_block: List[int], geom: Dict[str, Any], use_bn: bool = True):
        super(BackBone, self).__init__()

        self.use_bn = use_bn

        # Block 1
        self.conv1 = conv3x3(geom["input_shape"][-1], 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Block 2-5
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        p = 0 if geom["label_shape"][1] == 175 else 1
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input through bottom-up pathway.

        Parameters
        ----------
        x : torch.Tensor
            Input BEV torch.Tensor.

        Returns
        -------
        tuple of torch.Tensor
            Multi-scale features (c3, c4, c5) from different resolution levels.
        """
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        return c3, c4, c5

    def decode(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        """
        Decode multi-scale features through top-down pathway.

        Parameters
        ----------
        c3 : torch.Tensor
            Features from layer 3.
        c4 : torch.Tensor
            Features from layer 4.
        c5 : torch.Tensor
            Features from layer 5.

        Returns
        -------
        torch.Tensor
            Fused multi-scale features.
        """
        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)

        return p4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input BEV torch.Tensor with shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Multi-scale fused features with shape (N, 96, H/4, W/4).
        """
        c3, c4, c5 = self.encode(x)
        p4 = self.decode(c3, c4, c5)

        return p4

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, num_blocks: int) -> nn.Sequential:
        """
        Construct a residual layer with multiple blocks.

        Parameters
        ----------
        block : type
            Residual block type (BasicBlock or Bottleneck).
        planes : int
            Number of output channels.
        num_blocks : int
            Number of blocks in this layer.

        Returns
        -------
        nn.Sequential
            Sequential container of residual blocks.
        """
        if self.use_bn:
            # downsample the H*W by 1/2
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=2, bias=True)

        layers = [block(self.in_planes, planes, stride=2, downsample=downsample)]

        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class Header(nn.Module):
    """
    Prediction header for PIXOR.

    This header takes backbone features and produces classification and
    regression outputs for object detection.

    Parameters
    ----------
    use_bn : bool, optional
        Whether to use batch normalization. Default is True.

    Attributes
    ----------
    use_bn : bool
        Flag indicating whether batch normalization is used.
    conv1 : nn.Conv2d
        First 3x3 convolution layer.
    bn1 : nn.BatchNorm2d
        First batch normalization layer.
    conv2 : nn.Conv2d
        Second 3x3 convolution layer.
    bn2 : nn.BatchNorm2d
        Second batch normalization layer.
    conv3 : nn.Conv2d
        Third 3x3 convolution layer.
    bn3 : nn.BatchNorm2d
        Third batch normalization layer.
    conv4 : nn.Conv2d
        Fourth 3x3 convolution layer.
    bn4 : nn.BatchNorm2d
        Fourth batch normalization layer.
    clshead : nn.Conv2d
        Classification head producing object confidence scores.
    reghead : nn.Conv2d
        Regression head producing bounding box parameters.
    """

    def __init__(self, use_bn: bool = True):
        super(Header, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn
        self.conv1 = conv3x3(96, 96, bias=bias)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, 6, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through prediction header.

        Parameters
        ----------
        x : torch.Tensor
            Input features with shape (N, 96, H, W).

        Returns
        -------
        tuple of torch.Tensor
            Classification scores with shape (N, 1, H, W) and
            regression parameters with shape (N, 6, H, W).
        """
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = self.clshead(x)
        reg = self.reghead(x)

        return cls, reg


class PIXOR(nn.Module):
    """
    The Pixor backbone. The input of PIXOR nn module is a torch.Tensor of
    [batch_size, height, weight, channel], The output of PIXOR nn module
    is also a torch.Tensor of [batch_size, height/4, weight/4, channel].  Note that
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

    def __init__(self, args: Dict[str, Any]):
        super(PIXOR, self).__init__()
        geom = args["geometry_param"]
        use_bn = args["use_bn"]
        self.backbone = BackBone(Bottleneck, [3, 6, 6, 3], geom, use_bn)
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
        Forward pass through PIXOR model.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing:
            - 'processed_lidar': Dictionary with 'bev_input' BEV representation.

        Returns
        -------
        dict of str to torch.Tensor
        """
        bev_input = data_dict["processed_lidar"]["bev_input"]

        features = self.backbone(bev_input)
        # cls -- (N, 1, W/4, L/4)
        # reg -- (N, 6, W/4, L/4)
        cls, reg = self.header(features)

        output_dict = {"cls": cls, "reg": reg}

        return output_dict
