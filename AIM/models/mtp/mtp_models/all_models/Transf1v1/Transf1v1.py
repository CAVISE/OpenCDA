import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from AIM.models.mtp.mtp_models.all_models.encoders_car.transformer_based.encoder_car_v1 import CarsEncoder, SimpleDecoder


class Transf1v1(nn.Module, PyTorchModelHubMixin):
    """
    simple transformer model with cars encoder and decoder
    """

    def __init__(
        self, hidden_channels: int, n_heads: int, dropout: float, n_attn: int, bias: bool, n_linear_encoder: int = 2, n_linear_decoder: int = 2
    ) -> None:
        """
        initialize transformer model

        :param hidden_channels: hidden channels for encoder and decoder
        :param n_heads: number of attention heads
        :param dropout: dropout probability
        :param n_attn: number of self-attention blocks
        :param bias: whether to use bias in linear layers
        :param n_linear_encoder: number of linear layers in encoder
        :param n_linear_decoder: number of linear layers in decoder
        """
        super().__init__()
        self.cars_encoder = CarsEncoder(hidden_channels, n_heads, dropout, n_attn, bias, n_linear_encoder)
        self.simple_decoder = SimpleDecoder(hidden_channels, bias, n_linear_decoder)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        forward pass through transformer model

        :param x: input tensor with car features
        :param attn_mask: attention mask

        :return: predicted trajectories
        """
        x = self.cars_encoder(x, attn_mask)
        x = self.simple_decoder(x)
        return x
