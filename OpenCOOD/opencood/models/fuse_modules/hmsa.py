"""
Hierarchical Graph Transformer CAV Attention module.

This module implements a hierarchical multi-head attention mechanism for handling
heterogeneous agents with different types and relationships in cooperative perception.
"""

import torch
from torch import nn

from einops import rearrange

from typing import Tuple
from torch import Tensor


class HGTCavAttention(nn.Module):
    """
    Hierarchical Graph Transformer CAV Attention module.

    This module implements a hierarchical multi-head self-attention mechanism that handles
    different types of agents and their relationships.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    heads : int
        Number of attention heads.
    num_types : int, optional
        Number of different agent types. Default is 2.
    num_relations : int, optional
        Number of possible relation types (num_types * num_types). Default is 4.
    dim_head : int, optional
        Dimension of each attention head. Default is 64.
    dropout : float, optional
        Dropout probability. Default is 0.1.

     Attributes
    ----------
    heads : int
        Number of attention heads.
    scale : float
        Scaling factor for attention scores (1/sqrt(dim_head)).
    num_types : int
        Number of different agent types.
    attend : nn.Softmax
        Softmax layer for computing attention weights.
    drop_out : nn.Dropout
        Dropout layer for regularization.
    k_linears : nn.ModuleList
        List of linear layers for key projections, one per agent type.
    q_linears : nn.ModuleList
        List of linear layers for query projections, one per agent type.
    v_linears : nn.ModuleList
        List of linear layers for value projections, one per agent type.
    a_linears : nn.ModuleList
        List of linear layers for output projections, one per agent type.
    norms : nn.ModuleList
        List of normalization layers (currently unused).
    relation_att : nn.Parameter
        Learnable relation-specific attention parameters with shape
        (num_relations, heads, dim_head, dim_head).
    relation_msg : nn.Parameter
        Learnable relation-specific message parameters with shape
        (num_relations, heads, dim_head, dim_head).
    """

    def __init__(self, dim: int, heads: int, num_types: int = 2, num_relations: int = 4, dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head**-0.5
        self.num_types = num_types

        self.attend = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(dropout)
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()  # noqa: DC05
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, inner_dim))
            self.q_linears.append(nn.Linear(dim, inner_dim))
            self.v_linears.append(nn.Linear(dim, inner_dim))
            self.a_linears.append(nn.Linear(inner_dim, dim))

        self.relation_att = nn.Parameter(torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, heads, dim_head, dim_head))

        torch.nn.init.xavier_uniform(self.relation_att)
        torch.nn.init.xavier_uniform(self.relation_msg)

    def to_qkv(self, x: Tensor, types: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Transform input features into query, key, and value tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, L, C) where:

            - B : Batch size
            - H : Height
            - W : Width
            - L : Sequence length (number of agents)
            - C : Feature dimension
        types : torch.Tensor
            Agent type indices of shape (B, L).

        Returns
        -------
        q : torch.Tensor
            Query tensor of shape (B, H, W, L, inner_dim).
        k : torch.Tensor
            Key tensor of shape (B, H, W, L, inner_dim).
        v : torch.Tensor
            Value tensor of shape (B, H, W, L, inner_dim).
        """
        # x: (B,H,W,L,C)
        # types: (B,L)
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(x.shape[0]):
            q_list = []
            k_list = []
            v_list = []

            for i in range(x.shape[-2]):
                type_idx = int(types[b, i].item())
                # (H,W,1,C)
                q_list.append(self.q_linears[type_idx](x[b, :, :, i, :].unsqueeze(2)))
                k_list.append(self.k_linears[type_idx](x[b, :, :, i, :].unsqueeze(2)))
                v_list.append(self.v_linears[type_idx](x[b, :, :, i, :].unsqueeze(2)))
            # (1,H,W,L,C)
            q_batch.append(torch.cat(q_list, dim=2).unsqueeze(0))
            k_batch.append(torch.cat(k_list, dim=2).unsqueeze(0))
            v_batch.append(torch.cat(v_list, dim=2).unsqueeze(0))
        # (B,H,W,L,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)
        return q, k, v

    def get_relation_type_index(self, type1: Tensor, type2: Tensor) -> Tensor:
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x: Tensor, types: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute relation-specific attention and message weights.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, L, C).
        types : torch.Tensor
            Agent type indices of shape (B, L).

        Returns
        -------
        w_att : torch.Tensor
            Attention weights of shape (B, M, L, L, C_head, C_head).
        w_msg : torch.Tensor
            Message weights of shape (B, M, L, L, C_head, C_head).
        """
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i], types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def to_out(self, x: Tensor, types: Tensor) -> Tensor:
        """
        Project the attention output back to the original dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, H, W, L, inner_dim).
        types : torch.Tensor
            Agent type indices of shape (B, L).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, H, W, L, C).
        """
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[-2]):
                type_idx = int(types[b, i].item())
                out_list.append(self.a_linears[type_idx](x[b, :, :, i, :].unsqueeze(2)))
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))
        out = torch.cat(out_batch, dim=0)
        return out

    def forward(self, x: Tensor, mask: Tensor, prior_encoding: Tensor) -> Tensor:
        """
        Forward pass of the HGTCavAttention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, H, W, C).
        mask : torch.Tensor
            Attention mask of shape (B, H, W, L, 1).
        prior_encoding : torch.Tensor
            Prior encoding information of shape (B, L, H, W, 3).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, H, W, C).
        """
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, H, W, L, 1)
        # prior_encoding: (B,L,H,W,3)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)
        # (B,L)
        _, dts, types = [itm.squeeze(-1) for itm in prior_encoding[:, :, 0, 0, :].split([1, 1, 1], dim=-1)]
        types = types.to(torch.int)
        dts = dts.to(torch.int)
        qkv = self.to_qkv(x, types)
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, "b h w l (m c) -> b m h w l c", m=self.heads), (qkv))
        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum("b m h w i p, b m i j p q, bm h w j q -> b m h w i j", [q, w_att, k]) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float("inf"))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        v_msg = torch.einsum("b m i j p c, b m h w j p -> b m h w i j c", w_msg, v)
        out = torch.einsum("b m h w i j, b m h w i j c -> b m h w i c", att_map, v_msg)

        out = rearrange(out, "b m h w l c -> b h w l (m c)", m=self.heads)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out
