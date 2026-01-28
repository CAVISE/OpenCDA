"""
Swap Fusion (Fused Axial Attention) for multi-agent feature fusion.

This module implements swap fusion using alternating window and grid attention
patterns for efficient multi-agent cooperative perception.
"""

import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.sub_modules.base_transformer import FeedForward, PreNormResidual

from typing import Dict, Optional


# swap attention -> max_vit
class Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.

    Attributes
    ----------
    heads : int
        Number of attention heads.
    scale : float
        Scaling factor for queries (1/sqrt(dim_head)).
    window_size : list of int
        Window size in [agent, height, width] dimensions.
    to_qkv : nn.Linear
        Linear projection for queries, keys, and values.
    attend : nn.Sequential
        Softmax layer for attention weights.
    to_out : nn.Sequential
        Output projection with dropout.
    relative_position_bias_table : nn.Embedding
        Learnable relative position bias table.
    relative_position_index : Tensor
        Buffer storing relative position indices for bias lookup.
    """

    def __init__(self, dim: int, dim_head: int = 32, dropout: float = 0.0, agent_size: int = 6, window_size: int = 7):
        super().__init__()
        assert (dim % dim_head) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1))

        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), self.heads
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention with optional masking.

        Parameters
        ----------
        x : Tensor
            Input features (B, L, H, W, w_h, w_w, C).
        mask : Tensor, optional
            Binary mask (B, H, W, 1, L) for valid agents.

        Returns
        -------
        out : Tensor
            Attention output with same shape as input.
        """
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, _, h = *x.shape, x.device, self.heads  # eighth variable is device # NOTE: This unpacking relies on dynamic tensor shape and device injection.Precise typing is impossible without refactoring the assignment.

        # flatten
        x = rearrange(x, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, "i j h -> h i j")

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, "b x y w1 w2 e l -> (b x y) e (l w1 w2)")
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float("inf"))

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # merge heads
        out = rearrange(out, "b h (l w1 w2) d -> b l w1 w2 (h d)", l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, "(b x y) l w1 w2 d -> b l x y w1 w2 d", b=batch, x=height, y=width)


class SwapFusionBlockMask(nn.Module):
    """
    Swap fusion block with mask support for variable agent counts.

    Alternates between window attention (local) and grid attention (global).

    Parameters
    ----------
    input_dim : int
        Feature dimension.
    mlp_dim : int
        MLP hidden dimension.
    dim_head : int
        Dimension per attention head.
    window_size : int
        Window size for partitioning.
    agent_size : int
        Number of agents.
    drop_out : float
        Dropout rate.

    Attributes
    ----------
    heads : int
        Number of attention heads.
    scale : float
        Scaling factor for queries (1/sqrt(dim_head)).
    window_size : list of int
        Window size in [agent, height, width] dimensions.
    to_qkv : nn.Linear
        Linear projection for queries, keys, and values.
    attend : nn.Sequential
        Softmax layer for attention weights.
    to_out : nn.Sequential
        Output projection with dropout.
    relative_position_bias_table : nn.Embedding
        Learnable relative position bias table.
    relative_position_index : Tensor
        Buffer storing relative position indices for bias lookup.
    """

    def __init__(self, input_dim: int, mlp_dim: int, dim_head: int, window_size: int, agent_size: int, drop_out: float):
        super(SwapFusionBlockMask, self).__init__()

        self.window_size = window_size

        self.window_attention = PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out, agent_size, window_size))
        self.window_ffd = PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out))
        self.grid_attention = PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out, agent_size, window_size))
        self.grid_ffd = PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply window and grid attention with masking.

        Parameters
        ----------
        x : Tensor
            Input features (B, L, C, H, W).
        mask : Tensor
            Agent mask (B, H, W, 1, L).

        Returns
        -------
        x : Tensor
            Output features (B, L, C, H, W).
        """
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention
        mask_swap = mask

        # mask b h w 1 l -> b x y w1 w2 1 L
        mask_swap = rearrange(mask_swap, "b (x w1) (y w2) e l -> b x y w1 w2 e l", w1=self.window_size, w2=self.window_size)
        x = rearrange(x, "b m d (x w1) (y w2) -> b m x y w1 w2 d", w1=self.window_size, w2=self.window_size)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, "b m x y w1 w2 d -> b m d (x w1) (y w2)")

        # grid attention
        mask_swap = mask
        mask_swap = rearrange(mask_swap, "b (w1 x) (w2 y) e l -> b x y w1 w2 e l", w1=self.window_size, w2=self.window_size)
        x = rearrange(x, "b m d (w1 x) (w2 y) -> b m x y w1 w2 d", w1=self.window_size, w2=self.window_size)
        x = self.grid_attention(x, mask=mask_swap)
        x = self.grid_ffd(x)
        x = rearrange(x, "b m x y w1 w2 d -> b m d (w1 x) (w2 y)")

        return x


class SwapFusionBlock(nn.Module):
    """
    Swap fusion block without masking

     Parameters
     ----------
     input_dim : int
         Feature dimension.
     mlp_dim : int
         MLP hidden dimension.
     dim_head : int
         Dimension per attention head.
     window_size : int
         Window size for partitioning.
     agent_size : int
         Number of agents.
     drop_out : float
         Dropout rate.

     Attributes
     ----------
     block : nn.Sequential
         Sequential block containing window and grid attention operations.
    """

    def __init__(self, input_dim: int, mlp_dim: int, dim_head: int, window_size: int, agent_size: int, drop_out: float) -> None:
        super(SwapFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            Rearrange("b m d (x w1) (y w2) -> b m x y w1 w2 d", w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out, agent_size, window_size)),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b m x y w1 w2 d -> b m d (x w1) (y w2)"),
            Rearrange("b m d (w1 x) (w2 y) -> b m x y w1 w2 d", w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out, agent_size, window_size)),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b m x y w1 w2 d -> b m d (w1 x) (w2 y)"),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # todo: add mask operation later for mulit-agents
        x = self.block(x)
        return x


class SwapFusionEncoder(nn.Module):
    """
    Multi-stage swap fusion encoder with MLP head.

    Stacks multiple swap fusion blocks and averages across agents.

    Parameters
    ----------
    args : dict
        Configuration containing:
            - depth : int
                Number of swap fusion blocks.
            - input_dim : int
                Feature dimension.
            - mlp_dim : int
                MLP hidden dimension.
            - agent_size : int
                Number of agents.
            - window_size : int
                Window size.
            - dim_head : int
                Head dimension.
            - drop_out : float
                Dropout rate.
            - mask : bool, optional
                Whether to use masking.

    Attributes
    ----------
    layers : nn.ModuleList
        List of swap fusion blocks.
    depth : int
        Number of fusion blocks.
    mask : bool
        Flag indicating whether masking is enabled.
    mlp_head : nn.Sequential
        MLP head for final feature transformation after agent aggregation.
    """

    def __init__(self, args: Dict):
        super(SwapFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["depth"]

        # block related
        input_dim = args["input_dim"]
        mlp_dim = args["mlp_dim"]
        agent_size = args["agent_size"]
        window_size = args["window_size"]
        drop_out = args["drop_out"]
        dim_head = args["dim_head"]

        self.mask = False
        if "mask" in args:
            self.mask = args["mask"]

        for i in range(self.depth):
            if self.mask:
                block = SwapFusionBlockMask(input_dim, mlp_dim, dim_head, window_size, agent_size, drop_out)

            else:
                block = SwapFusionBlock(input_dim, mlp_dim, dim_head, window_size, agent_size, drop_out)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Reduce("b m d h w -> b d h w", "mean"),
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply swap fusion and aggregate across agents.

        Parameters
        ----------
        x : Tensor
            Input features (B, L, C, H, W).
        mask : Tensor, optional
            Agent mask (B, H, W, 1, L).

        Returns
        -------
        out : Tensor
            Fused features (B, C, H, W).
        """
        for stage in self.layers:
            x = stage(x, mask=mask)
        return self.mlp_head(x)


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = {"input_dim": 512, "mlp_dim": 512, "agent_size": 4, "window_size": 8, "dim_head": 4, "drop_out": 0.1, "depth": 2, "mask": True}
    block = SwapFusionEncoder(args)
    block.cuda()
    test_data = torch.rand(1, 4, 512, 32, 32)
    test_data = test_data.cuda()
    mask = torch.ones(1, 32, 32, 1, 4)
    mask = mask.cuda()

    output = block(test_data, mask)
    print(output)
