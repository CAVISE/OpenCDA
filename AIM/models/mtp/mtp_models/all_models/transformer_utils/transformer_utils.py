import torch
import torch.nn as nn
import torch.nn.functional as F


# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
# https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        self.bias = bias

        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=self.bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=self.bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=self.bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=self.bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=self.bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None (dtype = torch.bool), False - not use vector of sequence
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)

        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=is_causal)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class SelfAttnBlock(nn.Module):
    def __init__(self, hidden_channels, n_heads, dropout, bias, n_linear):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, hidden_channels, hidden_channels * n_heads, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.fc1 = nn.Linear(hidden_channels, 4 * hidden_channels, bias=bias)
        self.fc2 = nn.Linear(4 * hidden_channels, hidden_channels, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x, attn_mask):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.attn(x, x, x, attn_mask=attn_mask))

        residual = x
        x = self.norm2(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        x = residual + x
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, x_hidden_channels, y_hidden_channels, n_heads, dropout, bias, n_linear):
        super().__init__()
        self.attn = MultiHeadAttention(x_hidden_channels, y_hidden_channels, y_hidden_channels, x_hidden_channels * n_heads, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(x_hidden_channels)
        self.norm2 = nn.LayerNorm(x_hidden_channels)

        self.fc1 = nn.Linear(x_hidden_channels, 4 * x_hidden_channels, bias=bias)
        self.fc2 = nn.Linear(4 * x_hidden_channels, x_hidden_channels, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x, y, attn_mask):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.attn(x, y, y, attn_mask=attn_mask))

        residual = x
        x = self.norm2(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        x = residual + x
        return x
