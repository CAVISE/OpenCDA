import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from AIM.models.mtp.mtp_models.all_models.encoders_car.transformer_based.encoder_car_v1 import CarsEncoder, SimpleDecoder
from AIM.models.mtp.mtp_models.all_models.encoders_map.lane_lv_map_encoder.lane_lv_map_encoder import LaneMapEncoder
from AIM.models.mtp.mtp_models.all_models.transformer_utils.transformer_utils import CrossAttnBlock


class TransfAny_v1(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        map_encoder_base_channels=16,
        map_encoder_stages=(3, 4, 4, 3, 3),
        map_encoder_out_vec_size=128,
        cars_encoder_hidden_channels=128,
        cars_encoder_n_heads=4,
        dropout=0.2,
        cars_encoder_n_attn=4,
        bias=True,
        cars_encoder_n_linear_encoder=2,
        decoder_n_linear_decoder=2,
        cross_attn_n_heads=4,
        cross_attn_n_linear=2,
        cross_attn_n_attn=2,
    ):
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

    def forward(self, x, map, cars_attn_mask, map_attn_mask):
        x = self.cars_encoder(x, cars_attn_mask)
        map = self.map_encoder(map)

        cross_attn_mask = cars_attn_mask[:, :, 0:1] * map_attn_mask[:, 0:1, :]
        for cross_attn in self.cross_attns:
            x = cross_attn(x, map, cross_attn_mask)

        x = self.simple_decoder(x)
        return x
