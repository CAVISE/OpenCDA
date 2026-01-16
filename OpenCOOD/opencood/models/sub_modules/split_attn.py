"""
Split Attention module for multi-scale feature fusion.

This module implements split attention mechanism that adaptively weights
features from multiple window sizes using radix softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadixSoftmax(nn.Module):
    """
    Radix Softmax module that applies softmax along the radix dimension.
    
    Parameters
    ----------
    radix : int
        Number of splits (radix) for the input.
    cardinality : int
        Number of groups for grouped convolution.
    
    Attributes
    ----------
    radix : int
        Number of splits.
    cardinality : int
        Number of groups.
    """
    
    def __init__(self, radix: int, cardinality: int):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        """
        Apply radix softmax to compute split attention weights.

        Parameters
        ----------
        x : Tensor
            Input features with shape.

        Returns
        -------
        Tensor
            Attention weights with shape (B, radix*L*C) if radix > 1,
            or sigmoid activations with same shape as input if radix = 1.
        """
        # x: (B, L, 1, 1, 3C)
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(batch, cav_num, self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=3)
            # B, 3LC
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """
    Split Attention module for adaptive multi-scale feature fusion.

    This module fuses features from multiple window sizes (small, medium, big)
    using learned attention weights computed via radix softmax.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.

    Attributes
    ----------
    input_dim : int
        Feature dimension.
    fc1 : nn.Linear
        First fully connected layer for attention computation.
    bn1 : nn.LayerNorm
        Layer normalization after first FC layer.
    act1 : nn.ReLU
        ReLU activation.
    fc2 : nn.Linear
        Second fully connected layer producing attention logits.
    rsoftmax : RadixSoftmax
        Radix softmax module for computing attention weights.
    """
    
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
        """
        Fuse multi-scale window features using split attention.

        Parameters
        ----------
        window_list : list of Tensor
            List of 3 window features [small, medium, big], each with
            shape (B, L, H, W, C).

        Returns
        -------
        Tensor
            Attention-weighted fused features with shape (B, L, H, W, C).
        """
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 3, "only 3 windows are supported"

        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        B, L = sw.shape[0], sw.shape[1]

        # global average pooling, B, L, H, W, C
        x_gap = sw + mw + bw
        # B, L, 1, 1, C
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # B L 1 1 3C
        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)

        out = (
            sw * x_attn[:, :, :, :, 0 : self.input_dim]
            + mw * x_attn[:, :, :, :, self.input_dim : 2 * self.input_dim]
            + bw * x_attn[:, :, :, :, self.input_dim * 2 :]
        )

        return out
