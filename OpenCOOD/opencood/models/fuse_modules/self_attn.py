"""
Scaled Dot-Product Attention for multi-agent feature fusion.

This module implements scaled dot-product attention mechanism for aggregating
features from multiple agents in cooperative perception systems.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Computes attention weights by taking the dot product of queries with keys,
    scaling by sqrt(dim), applying softmax, and aggregating values.

    Parameters
    ----------
    dim : int
        Dimension of attention features.

    Attributes
    ----------
    sqrt_dim : float
        Square root of dimension for scaling attention scores.
    """

    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Apply scaled dot-product attention.

        Parameters
        ----------
        query : Tensor
            Query tensor with shape (batch, q_len, d_model).
        key : Tensor
            Key tensor with shape (batch, k_len, d_model).
        value : Tensor
            Value tensor with shape (batch, v_len, d_model).

        Returns
        -------
        Tensor
            Context vector from attention mechanism with shape (batch, q_len, d_model).
        """
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    """
    Attention-based fusion module for multi-agent feature aggregation.

    Applies scaled dot-product attention across CAV features at each spatial
    location to fuse information from multiple agents into ego vehicle's view.

    Parameters
    ----------
    feature_dim : int
        Feature channel dimension for attention computation.

    Attributes
    ----------
    att : ScaledDotProductAttention
        Self-attention module for cross-agent feature fusion.
    """

    def __init__(self, feature_dim: int):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x: torch.Tensor, record_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention-based fusion.

        Parameters
        ----------
        x : Tensor
            Input features from all agents with shape (sum(n_cav), C, H, W).
        record_len : Tensor
            Number of agents per batch sample with shape (B,).

        Returns
        -------
        Tensor
            Fused ego vehicle features with shape (B, C, H, W).
        """
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            out.append(h)
        return torch.stack(out)

    def regroup(self, x: torch.Tensor, record_len: torch.Tensor) -> List[torch.Tensor]:
        """
        Split input tensor into a list of tensors based on record_len.

        Parameters
        ----------
        x : Tensor
            Input tensor to be split with shape (sum(n_cav), C, H, W).
        record_len : Tensor
            Number of agents per sample with shape (B,).

        Returns
        -------
        list of Tensor
            List of tensors [(L1, C, H, W), (L2, C, H, W), ...], one per sample.
        """
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
