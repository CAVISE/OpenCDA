import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from AIM.models.mtp.mtp_models.all_models.encoders_car.transformer_based.encoder_car_v1 import CarsEncoder, SimpleDecoder


class Transf1v1(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_channels, n_heads, dropout, n_attn, bias, n_linear_encoder: int = 2, n_linear_decoder: int = 2):
        super().__init__()
        self.cars_encoder = CarsEncoder(hidden_channels, n_heads, dropout, n_attn, bias, n_linear_encoder)
        self.simple_decoder = SimpleDecoder(hidden_channels, bias, n_linear_decoder)

    def forward(self, x, attn_mask):
        x = self.cars_encoder(x, attn_mask)
        x = self.simple_decoder(x, attn_mask)
        return x
