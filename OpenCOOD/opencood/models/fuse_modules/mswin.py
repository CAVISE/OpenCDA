"""
Base Window Attention module for local self-attention.

This module implements window-based attention with optional relative position embeddings
for efficient processing of spatial features.
"""

import torch
import torch.nn as nn

import numpy as np

from einops import rearrange
from opencood.models.sub_modules.split_attn import SplitAttn

from typing import List


def get_relative_distances(window_size: int) -> torch.Tensor:
    """
    Generate relative position indices for a square window.

    Computes pairwise relative positions between all spatial locations
    in a square window, used for relative position encoding in attention.

    Parameters
    ----------
    window_size : int
        Size of the square window (height and width).

    Returns
    -------
    distances : torch.Tensor
        Relative position indices with shape (window_size^2, window_size^2, 2).
        distances[i, j, 0] is the relative row distance from position i to j.
        distances[i, j, 1] is the relative column distance from position i to j.
    """
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    """
    Base window attention module that applies self-attention within local windows.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension of each attention head.
    drop_out : float
        Dropout probability.
    window_size : int
        Size of the attention window.
    relative_pos_embedding : bool
        Whether to use relative position embeddings.

    Attributes
    ----------
    heads : int
        Number of attention heads.
    scale : float
        Scaling factor for attention scores (1/sqrt(dim_head)).
    window_size : int
        Size of the attention window.
    relative_pos_embedding : bool
        Flag indicating whether relative position embeddings are used.
    to_qkv : nn.Linear
        Linear projection for queries, keys, and values.
    relative_indices : Tensor, optional
        Buffer storing relative position indices if relative_pos_embedding is True.
    pos_embedding : nn.Parameter
        Learnable position embedding parameters.
    to_out : nn.Sequential
        Output projection with dropout.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, drop_out: float, window_size: int, relative_pos_embedding: bool):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(drop_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the base window attention.

        This method implements local window-based multi-head attention with optional
        relative positional embeddings. The input feature map is divided into
        non-overlapping windows, and attention is computed within each window.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, L, H, W, C) where:
            - B: batch size
            - L: sequence length (e.g., number of CAVs or time steps)
            - H: height of feature map
            - W: width of feature map
            - C: number of channels

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input (B, L, H, W, C) after
        applying window-based multi-head attention.
        """
        _, _, h, w, _, m = *x.shape, self.heads  # 1 -> b, 2 -> length, 5 -> c # NOTE: This unpacking relies on dynamic tensor shape and device injection.Precise typing is impossible without refactoring the assignment.

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        new_h = h // self.window_size
        new_w = w // self.window_size

        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        q, k, v = map(
            lambda t: rearrange(
                t, "b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c", m=m, w_h=self.window_size, w_w=self.window_size
            ),
            qkv,
        )
        # b l m h window_size window_size
        dots = (
            torch.einsum(
                "b l m h i c, b l m h j c -> b l m h i j",
                q,
                k,
            )
            * self.scale
        )
        # consider prior knowledge of the local window
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = torch.einsum("b l m h i j, b l m h j c -> b l m h i c", attn, v)
        # b l h w c
        out = rearrange(
            out,
            "b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)",
            m=self.heads,
            w_h=self.window_size,
            w_w=self.window_size,
            new_w=new_w,
            new_h=new_h,
        )
        out = self.to_out(out)

        return out


class PyramidWindowAttention(nn.Module):
    """
    Multi-scale window attention with multiple window sizes.

    Applies window-based self-attention at different scales (window sizes)
    and fuses the results using naive averaging or split attention.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    heads : list of int
        Number of attention heads for each window size.
    dim_heads : list of int
        Head dimension for each window size.
    drop_out : float
        Dropout probability.
    window_size : list of int
        Window sizes for multi-scale attention
    relative_pos_embedding : bool
        Whether to use relative position embeddings.
    fuse_method : str, optional
        Fusion method: 'naive' (average) or 'split_attn'. Default is 'naive'.

    Attributes
    ----------
    pwmsa : nn.ModuleList
        List of BaseWindowAttention modules for each scale.
    fuse_mehod : str
        Fusion method being used.
    split_attn : SplitAttn, optional
        Split attention fusion module (if fuse_method='split_attn').
    """

    def __init__(
        self,
        dim: int,
        heads: List[int],
        dim_heads: List[int],
        drop_out: float,
        window_size: List[int],
        relative_pos_embedding: bool,
        fuse_method: str = "naive",
    ):
        super().__init__()

        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)

        self.pwmsa = nn.ModuleList([])

        for head, dim_head, ws in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim, head, dim_head, drop_out, ws, relative_pos_embedding))
        self.fuse_mehod = fuse_method
        if fuse_method == "split_attn":
            self.split_attn = SplitAttn(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale window attention and fuse results.

        Parameters
        ----------
        x : torch.Tensor
            Input features (B, C, H, W) or (B, H*W, C).

        Returns
        -------
        output : torch.Tensor
            Fused multi-scale attention output with same shape as input.
        """
        output = None
        # naive fusion will just sum up all window attention output and do a
        # mean
        if self.fuse_mehod == "naive":
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            return output / len(self.pwmsa)

        elif self.fuse_mehod == "split_attn":
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            return self.split_attn(window_list)
