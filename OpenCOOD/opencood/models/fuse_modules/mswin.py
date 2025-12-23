"""
Multi-scale window transformer implementation for feature fusion in multi-agent systems.
This module provides window-based attention mechanisms with support for multi-scale processing.
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
    Args:
        window_size: Size of the square window (height/width)
    Returns:
        Tensor of shape (window_size^2, window_size^2, 2) containing relative positions
    """
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    """
    Base window attention module that applies self-attention within local windows.
    Args:
        dim: Input feature dimension
        heads: Number of attention heads
        dim_head: Dimension of each attention head
        drop_out: Dropout probability
        window_size: Size of the attention window
        relative_pos_embedding: Whether to use relative position embeddings
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        drop_out: float,
        window_size: int,
        relative_pos_embedding: bool
    ) -> None:
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
        Args:
            x: Input tensor of shape (batch_size, seq_len, height, width, channels)
        Returns:
            Output tensor of same shape as input
        """
        _, _, h, w, _, m = *x.shape, self.heads  # 1 -> b, 2 -> length, 5 -> c

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
    def __init__(
        self,
        dim: int,
        heads: List[int],
        dim_heads: List[int],
        drop_out: float,
        window_size: List[int],
        relative_pos_embedding: bool,
        fuse_method: str = "naive"
    ) -> None:
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

    def forward(self, x):
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
