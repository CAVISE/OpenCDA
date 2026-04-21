import torch
import torch.nn as nn
from typing import Tuple

from AIM.models.mtp.mtp_models.encoders_car.transformer_based.encoder_car import SimpleDecoder, CarsEncoder
from AIM.models.mtp.mtp_models.transformer_utils.transformer_utils import CrossAttnBlock
from AIM.aim_model import MTPModel


class ConvBlock(nn.Module):
    """
    convolutional block with batch normalization and activation
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        """
        initialize convolutional block

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size for convolution
        :param stride: stride for convolution
        :param padding: padding for convolution
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through convolutional block

        :param x: input tensor

        :return: output tensor
        """
        return self.act(self.bn(self.conv(x)))


class LaneMapEncoder(nn.Module):
    """
    encoder for lane-level map features using convolutional blocks
    """

    def __init__(
        self,
        base_channels: int = 16,
        stages: tuple[int, ...] = (3, 4, 4, 3, 3),
        out_channels: int = 128,
    ) -> None:
        """
        initialize lane map encoder

        :param base_channels: base number of channels
        :param stages: tuple of stage depths
        :param out_channels: output vector size
        """
        super().__init__()
        self.stem = ConvBlock(1, base_channels, kernel_size=4, stride=4, padding=0)  # image_size /4; channels
        self.stages = nn.ModuleList()
        channels = base_channels

        for i, depth in enumerate(stages):
            for j in range(depth):
                if j == 0 and i > 0:
                    stage = ConvBlock(channels, channels * 2, kernel_size=2, stride=2, padding=0)  # image_size /2; channels *2
                    channels *= 2
                else:
                    stage = ConvBlock(channels, channels)

                self.stages.append(stage)

        self.out_channels = channels
        self.norm = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through lane map encoder

        :param x: input tensor of shape (b, n, k, k) where b is batch size, n is number of lanes, k is map size

        :return: encoded map features of shape (b, n, out_channels)
        """
        b, n, k1, k2 = x.shape
        assert k1 == k2

        x = x.reshape(b * n, 1, k1, k2)  # (b * n, 1, k, k)
        x = self.stem(x)  # (b * n, 1, k/4, k/4)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)

        x = self.pool(x)  # (b * n, p, 1, 1)
        x = x.flatten(1)  # (b * n, p)
        x = self.head(x)  # (b * n, out_channels)
        x = x.reshape(b, n, -1)  # (b, n, out_channels)

        return x


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
