import torch
from torch import nn

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import INPUT_VECTOR_SIZE, PRED_LEN, PREDICT_VECTOR_SIZE
from AIM.models.mtp.mtp_models.all_models.transformer_utils.transformer_utils import SelfAttnBlock


class CarsEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, n_heads, dropout, n_attn, bias, n_linear=2):
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

    def forward(self, x, attn_mask):
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
    def __init__(self, input_channels, bias, n_linear=2):
        super().__init__()
        self.mlp = nn.ModuleList([nn.Linear(input_channels, input_channels, bias=bias) for i in range(n_linear)])
        self.out = nn.Linear(input_channels, PRED_LEN * PREDICT_VECTOR_SIZE)
        self.act = nn.GELU()

    def forward(self, x):
        for linear in self.mlp:
            x = self.act(linear(x))

        x = self.out(x)
        return x
