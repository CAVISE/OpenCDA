"""
Implementation of F-cooper maxout fusing.
"""

import torch
import torch.nn as nn

from typing import List

class SpatialFusion(nn.Module):
    """
    Spatial Fusion module that performs maxout fusion across the batch dimension.
    For each sample in the batch, it takes multiple feature maps and performs
    element-wise max pooling across them.
    """
    def __init__(self):
        super(SpatialFusion, self).__init__()

    def regroup(
        self,
        x: torch.Tensor,
        record_len: torch.Tensor
    ) -> List[torch.Tensor]:
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

    def forward(
        self,
        x: torch.Tensor,
        record_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the spatial fusion module.
        """
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)
