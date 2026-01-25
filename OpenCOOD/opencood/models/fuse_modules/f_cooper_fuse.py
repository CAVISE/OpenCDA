"""
F-Cooper Maxout Fusion implementation.

This module implements the spatial maxout fusion strategy from F-Cooper,
which aggregates multi-agent features using element-wise max pooling.
"""

import torch
import torch.nn as nn

from typing import List


class SpatialFusion(nn.Module):
    """
    Spatial Fusion module using maxout fusion across agents.

    This module performs element-wise max pooling across multiple agent feature maps
    to create a fused representation. For each batch sample, it takes features from
    multiple agents and selects the maximum activation at each spatial location.
    """

    def __init__(self) -> None:
        super(SpatialFusion, self).__init__()

    def regroup(self, x: torch.Tensor, record_len: torch.Tensor) -> List[torch.Tensor]:
        """
        Split the input tensor into a list of tensors based on record_len.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [sum(record_len), C, H, W].
        record_len : torch.Tensor
            Number of features per sample in the batch, shape [B].

        Returns
        -------
        list of torch.Tensor
            List of tensors where each tensor has shape [N_i, C, H, W],
            where N_i is the number of features for sample i
        """
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x: torch.Tensor, record_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass performing maxout fusion across agents.

        Parameters
        ----------
        x : Tensor
            Input features from all agents with shape (sum(record_len), C, H, W).
        record_len : Tensor
            Number of agents per batch sample with shape (B,).

        Returns
        -------
        Tensor
            Fused features with shape (B, C, H, W) after max pooling across agents.
        """
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)
