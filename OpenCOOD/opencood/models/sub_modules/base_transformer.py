"""
Transformer Components for Collaborative Perception.

This module implements transformer-based attention mechanisms for fusing
features from multiple Connected and Autonomous Vehicles (CAVs).
"""

from typing import Any

import torch
from torch import nn

from einops import rearrange


class PreNormResidual(nn.Module):
    """
    Pre-normalization with residual connection.

    This module applies layer normalization before a function and adds
    a residual connection.

    Parameters
    ----------
    dim : int
        Feature dimension.
    fn : nn.Module
        Function to apply after normalization.

    Attributes
    ----------
    norm : nn.LayerNorm
        Layer normalization.
    fn : nn.Module
        Function module.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass with pre-normalization and residual connection.

        Parameters
        ----------
        x : Tensor
            Input features.
        **kwargs
            Additional keyword arguments passed to fn.

        Returns
        -------
        Tensor
            Output features with residual connection.
        """
        return self.fn(self.norm(x), **kwargs) + x


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper.

    This module applies layer normalization before a function without
    residual connection.

    Parameters
    ----------
    dim : int
        Feature dimension.
    fn : nn.Module
        Function to apply after normalization.

    Attributes
    ----------
    norm : nn.LayerNorm
        Layer normalization.
    fn : nn.Module
        Function module.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass with pre-normalization.

        Parameters
        ----------
        x : Tensor
            Input features.
        **kwargs
            Additional keyword arguments passed to fn.

        Returns
        -------
        Tensor
            Normalized and processed features.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    This module implements a two-layer MLP with GELU activation and dropout.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float, optional
        Dropout probability. Default is 0.0.

    Attributes
    ----------
    net : nn.Sequential
        Sequential network of Linear-GELU-Dropout-Linear-Dropout.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Parameters
        ----------
        x : Tensor
            Input features.

        Returns
        -------
        Tensor
            Transformed features.
        """
        return self.net(x)


class CavAttention(nn.Module):
    """
    CAV (Connected and Autonomous Vehicle) multi-head attention.

    This module implements vanilla multi-head attention for fusing features
    from multiple CAVs with masking support.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    heads : int
        Number of attention heads.
    dim_head : int, optional
        Dimension per attention head. Default is 64.
    dropout : float, optional
        Dropout probability. Default is 0.1.

    Attributes
    ----------
    heads : int
        Number of attention heads.
    scale : float
        Scaling factor for attention scores (dim_head^-0.5).
    attend : nn.Softmax
        Softmax layer for attention weights.
    to_qkv : nn.Linear
        Linear layer to project input to queries, keys, and values.
    to_out : nn.Sequential
        Output projection with dropout.
    """

    def __init__(self, dim: int, heads: int, dim_head: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, prior_encoding: Any = None) -> torch.Tensor:
        """
        Forward pass through CAV attention.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, L, H, W, C) where L is number of CAVs.
        mask : Tensor
            CAV presence mask with shape (B, L) where 1 indicates valid CAV.
        prior_encoding : Any, optional
            Prior encoding (currently unused).

        Returns
        -------
        Tensor
            Attention output with shape (B, L, H, W, C).
        """
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, L)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)

        # qkv: [(B, H, W, L, C_inner) *3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, "b h w l (m c) -> b m h w l c", m=self.heads), qkv)

        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum("b m h w i c, b m h w j c -> b m h w i j", q, k) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float("inf"))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        out = torch.einsum("b m h w i j, b m h w j c -> b m h w i c", att_map, v)
        out = rearrange(out, "b m h w l c -> b h w l (m c)", m=self.heads)
        out = self.to_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out
