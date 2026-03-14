import torch
from torch import nn

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import INPUT_VECTOR_SIZE, PRED_LEN, PREDICT_VECTOR_SIZE
from AIM.models.mtp.mtp_models.transformer_utils.transformer_utils import SelfAttnBlock


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

        self.linear1 = nn.Linear(INPUT_VECTOR_SIZE, hidden_channels // 2, bias=bias)
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
    simple decoder with mlp layers for trajectory prediction
    """

    def __init__(self, input_channels: int, bias: bool, n_linear: int = 2) -> None:
        """
        initialize simple decoder

        :param input_channels: number of input channels
        :param bias: whether to use bias in linear layers
        :param n_linear: number of linear layers in mlp
        """
        super().__init__()
        self.mlp = nn.ModuleList([nn.Linear(input_channels, input_channels, bias=bias) for i in range(n_linear)])
        self.out = nn.Linear(input_channels, PRED_LEN * PREDICT_VECTOR_SIZE)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through decoder

        :param x: input tensor with encoded features

        :return: predicted trajectories
        """
        for linear in self.mlp:
            x = self.act(linear(x))

        x = self.out(x)
        return x
