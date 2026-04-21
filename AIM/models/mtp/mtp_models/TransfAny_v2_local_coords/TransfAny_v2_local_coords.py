import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn.conv import HeteroConv, GCNConv

from AIM.models.mtp.mtp_models.encoders_car.transformer_based.encoder_car import SimpleDecoder
from AIM.models.mtp.mtp_models.transformer_utils.transformer_utils import CrossAttnBlock, SelfAttnBlock
from AIM.aim_model import MTPModel
from AIM.models.mtp.learning.learning_src.data_scripts.data_config import config


class Encoder_v2(torch.nn.Module):
    """
    encoder for car features
    """

    def __init__(self, input_channels: int, hidden_channels: int, bias: bool) -> None:
        """
        initialize cars encoder

        :param hidden_channels: number of hidden channels
        :param bias: whether to use bias in linear layers
        """
        super().__init__()
        assert hidden_channels % 2 == 0

        self.linear1 = nn.Linear(input_channels, hidden_channels // 2, bias=bias)
        self.linear2 = nn.Linear(hidden_channels // 2, hidden_channels, bias=bias)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return x


class LaneConv(torch.nn.Module):
    def __init__(
        self,
        lane_hidden_channels: int,
        lane_conv_iters: int = 4,
    ) -> None:
        super().__init__()
        self.lane_conv_iters = lane_conv_iters

        conv_keys = (
            ["left", "right"]
            + [f"successor_{2**i}" for i in range(config.model.k_dot_steps)]
            + [f"predicessor_{2**i}" for i in range(config.model.k_dot_steps)]
        )
        self.hetero_convs = nn.ModuleList()
        self.self_linears = nn.ModuleList()
        for i in range(self.lane_conv_iters):
            conv_dict = {}

            for conv_key in conv_keys:
                conv_dict[("dot", conv_key, "dot")] = GCNConv(lane_hidden_channels, lane_hidden_channels, normalize=False, bias=False)
            self.hetero_convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.self_linears.append(nn.Linear(lane_hidden_channels, lane_hidden_channels, bias=False))

        self.norms = nn.ModuleList([nn.LayerNorm(lane_hidden_channels) for i in range(self.lane_conv_iters)])
        self.linear = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(lane_hidden_channels, lane_hidden_channels, bias=False), nn.LayerNorm(lane_hidden_channels))
                for i in range(self.lane_conv_iters)
            ]
        )
        self.act = nn.GELU()

    def forward(self, map_infos: Batch) -> Batch:
        # map_infos["dot"].x.shape (n, features)

        res = map_infos["dot"].x

        for i in range(self.lane_conv_iters):
            map_infos["dot"].x = self.hetero_convs[i](map_infos.x_dict, map_infos.edge_index_dict)["dot"] + self.self_linears[i](map_infos["dot"].x)

            map_infos["dot"].x = self.act(self.norms[i](map_infos["dot"].x))
            map_infos["dot"].x = self.act(self.linear[i](map_infos["dot"].x))

            map_infos["dot"].x += res
            map_infos["dot"].x = self.act(map_infos["dot"].x)
            res = map_infos["dot"].x

        return map_infos["dot"].x


class LaneEncoder(torch.nn.Module):
    def __init__(
        self,
        lane_input_channels: int,
        lane_hidden_channels: int,
        bias: bool,
        lane_conv_iters: int = 4,
    ):
        super().__init__()

        self.encoder = Encoder_v2(lane_input_channels, lane_hidden_channels, bias)
        self.laneConv = LaneConv(lane_hidden_channels, lane_conv_iters)

    def forward(self, map_infos: Batch) -> Batch:
        # map_infos["dot"].x.shape (n, features)

        map_infos["dot"].x = self.encoder(map_infos["dot"].x)
        map_infos_x = self.laneConv(map_infos)

        return map_infos_x


class A2M_M2A_CrossAttn(torch.nn.Module):
    def __init__(
        self,
        x_hidden_channels: int,
        y_hidden_channels: int,
        cross_attn_n_heads: int,
        dropout: float,
        bias: bool,
        cross_attn_n_attn: int,
        cross_attn_n_linear: int = 2,
    ):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(x_hidden_channels, x_hidden_channels, bias=False), nn.LayerNorm(x_hidden_channels))
        self.cross_attns = nn.ModuleList(
            [
                CrossAttnBlock(x_hidden_channels, y_hidden_channels, cross_attn_n_heads, dropout, bias=bias, n_linear=cross_attn_n_linear)
                for i in range(cross_attn_n_attn)
            ]
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, y: torch.Tensor, cross_attn_mask: torch.Tensor):
        x = self.act(self.linear(x))

        for attn in self.cross_attns:
            x = attn(x, y, cross_attn_mask)

        return x


class A2A_SelfAttn(torch.nn.Module):
    def __init__(
        self,
        cars_hidden_channels: int,
        self_attn_n_heads: int,
        dropout: float,
        bias: bool,
        self_attn_n_attn: int,
        self_attn_n_linear: int = 2,
    ):
        super().__init__()

        self.self_attns = nn.ModuleList(
            [SelfAttnBlock(cars_hidden_channels, self_attn_n_heads, dropout, bias=bias, n_linear=self_attn_n_linear) for i in range(self_attn_n_attn)]
        )
        self.act = nn.GELU()

    def forward(self, car_embeddings: torch.Tensor, self_attn_mask: torch.Tensor):
        for attn in self.self_attns:
            car_embeddings = attn(car_embeddings, self_attn_mask)

        return car_embeddings


class TransfAny_v2_local_coords(MTPModel):
    """
    transformer model with map encoder and cross-attention between cars and map features
    """

    def __init__(
        self,
        cars_hidden_channels: int = 64,
        lane_hidden_channels: int = 64,
        lane_conv_iters: int = 4,
        bias: bool = True,
        dropout: float = 0.2,
        cross_attn_n_heads: int = 4,
        cross_attn_n_attn: int = 2,
        cross_attn_n_linear: int = 2,
        cars_self_attn_n_heads: int = 4,
        cars_self_attn_n_attn: int = 4,
        cars_self_attn_n_linear: int = 2,
        decoder_n_linear_decoder: int = 2,
    ) -> None:
        """
        initialize transformer model with map encoder

        :param cars_hidden_channels: hidden channels for cars encoder
        :param lane_hidden_channels: hidden channels for lane encoder
        :param lane_conv_iters: iters of lane convolution
        :param bias: whether to use bias in linear layers
        :param dropout: dropout probability
        :param cross_attn_n_heads: number of attention heads for cross-attention
        :param cross_attn_n_attn: number of cross-attention blocks
        :param cross_attn_n_linear: number of linear layers in cross-attention
        :param cars_self_attn_n_heads: number of self attention heads for cars
        :param cars_self_attn_n_attn: number of self attention blocks for cars
        :param cars_self_attn_n_linear: number of linear layers in self attention for cars
        :param decoder_n_linear_decoder: number of linear layers in decoder

        """

        super().__init__()
        self.cars_hidden_channels = cars_hidden_channels
        self.lane_hidden_channels = lane_hidden_channels

        self.cars_encoder = Encoder_v2(config.model.input_vector_size, cars_hidden_channels, bias)
        self.map_encoder = LaneEncoder(config.object_map.object_vector_size, lane_hidden_channels, bias, lane_conv_iters)

        self.m2m_laneConv = LaneConv(lane_hidden_channels, lane_conv_iters)
        self.a2m_crossAttn = A2M_M2A_CrossAttn(
            lane_hidden_channels, cars_hidden_channels, cross_attn_n_heads, dropout, bias, cross_attn_n_attn, cross_attn_n_linear
        )
        self.m2a_crossAttn = A2M_M2A_CrossAttn(
            cars_hidden_channels, lane_hidden_channels, cross_attn_n_heads, dropout, bias, cross_attn_n_attn, cross_attn_n_linear
        )
        self.a2a_selfAttn = A2A_SelfAttn(cars_hidden_channels, cars_self_attn_n_heads, dropout, bias, cars_self_attn_n_attn, cars_self_attn_n_linear)

        self.cars_decoder = SimpleDecoder(cars_hidden_channels, bias, decoder_n_linear_decoder)
        self.movement_classification_head = SimpleDecoder(cars_hidden_channels, bias, decoder_n_linear_decoder, out_vec_size=1)

    def forward(
        self, cars_x: torch.Tensor, map_infos: Batch, cars_attn_mask: torch.Tensor, map_attn_mask: torch.Tensor, new_map_infos_shape
    ) -> torch.Tensor:
        """
        forward pass through transformer model

        :param x: input tensor with car features
        :param map: input tensor with map features
        :param cars_attn_mask: attention mask for cars encoder
        :param map_attn_mask: attention mask for map encoder

        :return: predicted trajectories
        """
        # cars_x.shape() (batch, n_vec, n_vec, features)
        # cars_attn_mask.shape() (batch, n_vec, n_vec, n_vec)
        # map_attn_mask.shape() (batch, n_vec, max_lanes*n_dots)

        batch_size, n_vec1, n_vec2, n_feat = cars_x.shape
        map_infos_x = map_infos["dot"].x
        old_map_infos_shape = map_infos_x.shape

        vecs_dots_dists = torch.norm(cars_x.unsqueeze(3)[..., [0, 1]] - map_infos_x.view(new_map_infos_shape).unsqueeze(2)[..., [0, 1]], dim=-1)
        valid_mask = cars_attn_mask[:, 0, :, :].unsqueeze(-1) & map_attn_mask.unsqueeze(2)
        radius = config.object_map.vecs_dots_radius_factor * 2 / config.object_map.n_lane_samples
        cross_attn_mask = (vecs_dots_dists <= radius) & valid_mask  # cross attn mask with shape (batch, n_vec, n_vec, max_lanes*n_dots)

        # encoding data
        cars_x = self.cars_encoder(cars_x)
        map_infos_x = self.map_encoder(map_infos)

        # a2m cross attn
        cars_x = cars_x.view(batch_size * n_vec1, n_vec2, self.cars_hidden_channels)
        map_infos_x = map_infos_x.view(new_map_infos_shape[0] * new_map_infos_shape[1], new_map_infos_shape[2], self.lane_hidden_channels)
        cross_attn_mask = cross_attn_mask.view(batch_size * n_vec1, n_vec2, new_map_infos_shape[2])
        map_infos_x = self.a2m_crossAttn(map_infos_x, cars_x, cross_attn_mask.transpose(-1, -2))

        # m2m lane conv
        map_infos_x = map_infos_x.view(old_map_infos_shape[0], self.lane_hidden_channels)
        map_infos["dot"].x = map_infos_x
        map_infos_x = self.m2m_laneConv(map_infos)

        # m2a cross attn
        cars_x = cars_x.view(batch_size * n_vec1, n_vec2, self.cars_hidden_channels)
        map_infos_x = map_infos_x.view(new_map_infos_shape[0] * new_map_infos_shape[1], new_map_infos_shape[2], self.lane_hidden_channels)
        cross_attn_mask = cross_attn_mask.view(batch_size * n_vec1, n_vec2, new_map_infos_shape[2])
        cars_x = self.m2a_crossAttn(cars_x, map_infos_x, cross_attn_mask)

        # a2a self attn
        cars_x = cars_x.view(batch_size * n_vec1, n_vec2, self.cars_hidden_channels)
        self_attn_mask = cars_attn_mask.view(batch_size * n_vec1, n_vec2, n_vec2)
        cars_x = self.a2a_selfAttn(cars_x, self_attn_mask)

        cars_x = cars_x.view(batch_size, n_vec1, n_vec2, self.cars_hidden_channels)
        cars_x = cars_x.diagonal(dim1=1, dim2=2).permute(0, 2, 1)

        movement_classification = self.movement_classification_head(cars_x)
        cars_x = self.cars_decoder(cars_x)

        return movement_classification, cars_x
