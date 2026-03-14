import torch
import torch.nn as nn
from typing import Tuple

from AIM.models.mtp.mtp_models.encoders_car.transformer_based.encoder_car_v1 import CarsEncoder, SimpleDecoder
from AIM.models.mtp.mtp_models.encoders_map.lane_lv_map_encoder.lane_lv_map_encoder import LaneMapEncoder
from AIM.models.mtp.mtp_models.transformer_utils.transformer_utils import CrossAttnBlock
from AIM.aim_model import MTPModel


class TransfAny_v1(MTPModel):
    """
    transformer model with map encoder and cross-attention between cars and map features
    """

    def __init__(
        self,
        map_encoder_base_channels: int = 16,
        map_encoder_stages: Tuple[int, ...] = (3, 4, 4, 3, 3),
        map_encoder_out_vec_size: int = 128,
        cars_encoder_hidden_channels: int = 128,
        cars_encoder_n_heads: int = 4,
        dropout: float = 0.2,
        cars_encoder_n_attn: int = 4,
        bias: bool = True,
        cars_encoder_n_linear_encoder: int = 2,
        decoder_n_linear_decoder: int = 2,
        cross_attn_n_heads: int = 4,
        cross_attn_n_linear: int = 2,
        cross_attn_n_attn: int = 2,
    ) -> None:
        """
        initialize transformer model with map encoder

        :param map_encoder_base_channels: base number of channels for map encoder
        :param map_encoder_stages: tuple of stage depths for map encoder
        :param map_encoder_out_vec_size: output vector size for map encoder
        :param cars_encoder_hidden_channels: hidden channels for cars encoder
        :param cars_encoder_n_heads: number of attention heads for cars encoder
        :param dropout: dropout probability
        :param cars_encoder_n_attn: number of self-attention blocks in cars encoder
        :param bias: whether to use bias in linear layers
        :param cars_encoder_n_linear_encoder: number of linear layers in cars encoder
        :param decoder_n_linear_decoder: number of linear layers in decoder
        :param cross_attn_n_heads: number of attention heads for cross-attention
        :param cross_attn_n_linear: number of linear layers in cross-attention
        :param cross_attn_n_attn: number of cross-attention blocks
        """
        super().__init__()
        self.map_encoder = LaneMapEncoder(map_encoder_base_channels, map_encoder_stages, map_encoder_out_vec_size)
        self.cars_encoder = CarsEncoder(
            cars_encoder_hidden_channels, cars_encoder_n_heads, dropout, cars_encoder_n_attn, bias, cars_encoder_n_linear_encoder
        )
        self.cross_attns = nn.ModuleList(
            [
                CrossAttnBlock(
                    cars_encoder_hidden_channels, map_encoder_out_vec_size, cross_attn_n_heads, dropout, bias=bias, n_linear=cross_attn_n_linear
                )
                for i in range(cross_attn_n_attn)
            ]
        )
        self.simple_decoder = SimpleDecoder(cars_encoder_hidden_channels, bias, decoder_n_linear_decoder)

    def forward(self, x: torch.Tensor, map: torch.Tensor, cars_attn_mask: torch.Tensor, map_attn_mask: torch.Tensor) -> torch.Tensor:
        """
        forward pass through transformer model

        :param x: input tensor with car features
        :param map: input tensor with map features
        :param cars_attn_mask: attention mask for cars encoder
        :param map_attn_mask: attention mask for map encoder

        :return: predicted trajectories
        """
        x = self.cars_encoder(x, cars_attn_mask)
        map = self.map_encoder(map)

        cross_attn_mask = cars_attn_mask[:, :, 0:1] * map_attn_mask[:, 0:1, :]
        for cross_attn in self.cross_attns:
            x = cross_attn(x, map, cross_attn_mask)

        x = self.simple_decoder(x)
        return x
