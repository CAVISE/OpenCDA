import torch
from torch import nn

from AIM.models.mtp.mtp_models.transformer_utils.transformer_utils import SelfAttnBlock
from AIM.models.mtp.learning.learning_src.data_scripts.data_config import config


class CarsEncoder(torch.nn.Module):
    """
    encoder for car features using transformer self-attention blocks
    """

    def __init__(self, hidden_channels: int, n_heads: int, dropout: float, n_attn: int, bias: bool, n_linear: int = 2) -> None:
        """
        initialize cars encoder

        :param hidden_channels: number of hidden channels
        :param n_heads: number of attention heads
        :param dropout: dropout probability
        :param n_attn: number of self-attention blocks
        :param bias: whether to use bias in linear layers
        :param n_linear: number of linear layers in attention blocks
        """
        super().__init__()
        assert hidden_channels % 2 == 0

        self.linear1 = nn.Linear(config.model.input_vector_size, hidden_channels // 2, bias=bias)
        self.linear2 = nn.Linear(hidden_channels // 2, hidden_channels, bias=bias)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.attns = nn.ModuleList([SelfAttnBlock(hidden_channels, n_heads, dropout, bias=bias, n_linear=n_linear) for i in range(n_attn)])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        forward pass through cars encoder

        :param x: input tensor with car features
        :param attn_mask: attention mask

        :return: encoded car features
        """
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))

        res = x
        x = self.norm1(x)
        x = self.act(self.linear3(x)) + res

        res = x
        x = self.norm2(x)
        x = self.act(self.linear4(x)) + res

        for attn in self.attns:
            x = attn(x, attn_mask)
        return x


class SimpleDecoder(torch.nn.Module):
    """
    MLP decoder for trajectory prediction: optional widen (hidden_scale) then hidden blocks, then linear readout.
    """

    def __init__(
        self,
        input_channels: int,
        bias: bool,
        n_linear: int = 2,
        hidden_scale: int = 1,
        dropout: float = 0.0,
        out_vec_size: int = None,
    ) -> None:
        """
        :param input_channels: embedding width from encoder
        :param bias: use bias in linear layers
        :param n_linear: number of hidden Linear(hidden, hidden) blocks after the widen layer
        :param hidden_scale: widen factor (1 = same as legacy: all layers input_channels-wide)
        :param dropout: dropout after each hidden block (0 = off)
        """
        super().__init__()
        hidden = max(1, input_channels * hidden_scale)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

        if hidden != input_channels:
            self.in_proj = nn.Linear(input_channels, hidden, bias=bias)
            self.norm_in = nn.LayerNorm(hidden)
            self.norm_h = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_linear)])
        else:
            self.in_proj = None
            self.norm_in = None
            self.norm_h = None

        self.mlp = nn.ModuleList([nn.Linear(hidden, hidden, bias=bias) for _ in range(n_linear)])

        if out_vec_size is None:
            out_vec_size = config.model.predict_vector_size

        out_channels = config.model.pred_len * out_vec_size
        self.out = nn.Linear(hidden, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_proj is not None:
            x = self.act(self.norm_in(self.in_proj(x)))

        if self.norm_h is None:
            for linear in self.mlp:
                x = self.act(linear(x))
                if self.drop is not None:
                    x = self.drop(x)
        else:
            for linear, norm in zip(self.mlp, self.norm_h):
                x = self.act(norm(linear(x)))
                if self.drop is not None:
                    x = self.drop(x)

        x = self.out(x)
        return x
