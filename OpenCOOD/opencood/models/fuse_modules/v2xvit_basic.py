"""
V2X Transformer implementation for multi-agent feature fusion.

This module implements the V2X Transformer architecture for fusing features
from multiple agents using attention mechanisms and temporal encoding.
"""
from typing import Dict, List, Union, Any
import math

import torch
from torch import nn

from opencood.models.fuse_modules.hmsa import HGTCavAttention
from opencood.models.fuse_modules.mswin import PyramidWindowAttention
from opencood.models.sub_modules.base_transformer import CavAttention, FeedForward, PreNorm
from opencood.models.sub_modules.torch_transformation_utils import (
    get_discretized_transformation_matrix,
    get_roi_and_cav_mask,
    get_transformation_matrix,
    warp_affine,
)


class STTF(nn.Module):
    """
    Spatial-Temporal Transformation Fusion module.

    This module applies spatial transformations to align features from different
    agents across time and space using affine transformations.

    Parameters
    ----------
    args : dict of str to Union[float, list of float, int]
        Configuration dictionary containing:
        - 'voxel_size': Voxel size [x, y, z].
        - 'downsample_rate': Downsampling rate for features.

    Attributes
    ----------
    discrete_ratio : float
        Discretization ratio from voxel size.
    downsample_rate : int
        Feature downsampling rate.
    """
    
    def __init__(self, args: Dict[str, Union[float, List[float], int]]):
        super(STTF, self).__init__()
        self.discrete_ratio = args["voxel_size"][0]
        self.downsample_rate = args["downsample_rate"]

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        spatial_correction_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass applying spatial-temporal transformation.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, L, H, W, C).
        mask : Tensor
            Mask tensor for valid agents.
        spatial_correction_matrix : Tensor
            Spatial correction matrices for alignment.

        Returns
        -------
        Tensor
            Transformed features with shape (B, L, H, W, C).
        """
        x = x.permute(0, 1, 4, 2, 3)
        dist_correction_matrix = get_discretized_transformation_matrix(spatial_correction_matrix, self.discrete_ratio, self.downsample_rate)
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(dist_correction_matrix[:, 1:, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, 1:, :, :, :].reshape(-1, C, H, W), T, (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)
        x = torch.cat([x[:, 0, :, :, :].unsqueeze(1), cav_features], dim=1)
        x = x.permute(0, 1, 3, 4, 2)
        return x


class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    
    Parameters
    ----------
    n_hid : int
        Hidden dimension size.
    RTE_ratio : float
        Ratio for temporal encoding.
    max_len : int, optional
        Maximum sequence length. Default is 100.
    dropout : float, optional
        Dropout probability. Default is 0.2.

    Attributes
    ----------
    RTE_ratio : float
        Temporal encoding scaling ratio.
    emb : nn.Embedding
        Embedding layer with sinusoidal weights.
    lin : nn.Linear
        Linear projection layer.
    """

    def __init__(
        self, 
        n_hid: int, 
        RTE_ratio: float, 
        max_len: int = 100, 
        dropout: float = 0.2
    ) -> None:
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass adding temporal encoding.

        Parameters
        ----------
        x : Tensor
            Input features with shape (H, W, C).
        t : Tensor
            Time delay scalar.

        Returns
        -------
        Tensor
            Features with temporal encoding added, shape (H, W, C).
        """
        # When t has unit of 50ms, rte_ratio=1.
        # So we can train on 100ms but test on 50ms
        return x + self.lin(self.emb(t * self.RTE_ratio)).unsqueeze(0).unsqueeze(1)


class RTE(nn.Module):
    """
    Relative Temporal Encoding wrapper for batched processing.
    
    Parameters
    ----------
    dim : int
        Feature dimension.
    RTE_ratio : float, optional
        Ratio for temporal encoding. Default is 2.

    Attributes
    ----------
    RTE_ratio : float
        Temporal encoding scaling ratio.
    emb : RelTemporalEncoding
        Underlying temporal encoding module.
    """
    
    def __init__(self, dim: int, RTE_ratio: float = 2):
        super(RTE, self).__init__()
        self.RTE_ratio = RTE_ratio

        self.emb = RelTemporalEncoding(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for relative temporal encoding.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, L, H, W, C).
        dts : Tensor
            Time delays with shape (B, L).

        Returns
        -------
        Tensor
            Features with temporal encoding added, shape (B, L, H, W, C).
        """
        # x: (B,L,H,W,C)
        # dts: (B,L)
        rte_batch = []
        for b in range(x.shape[0]):
            rte_list = []
            for i in range(x.shape[1]):
                rte_list.append(self.emb(x[b, i, :, :, :], dts[b, i]).unsqueeze(0))
            rte_batch.append(torch.cat(rte_list, dim=0).unsqueeze(0))
        return torch.cat(rte_batch, dim=0)


class V2XFusionBlock(nn.Module):
    """
    V2X Fusion Block combining multi-agent attention and pyramid window attention.
    
    Parameters
    ----------
    num_blocks : int
        Number of attention blocks.
    cav_att_config : Dict[str, Any]
        Configuration for CAV attention.
    pwindow_config : Dict[str, Any]
        Configuration for pyramid window attention.

    Attributes
    ----------
    layers : nn.ModuleList
        List of attention layer pairs (CAV attention + pyramid window attention).
    num_blocks : int
        Number of attention blocks.
    """
    
    def __init__(
        self, 
        num_blocks: int, 
        cav_att_config: Dict[str, Any],
        pwindow_config: Dict[str, Any]
    ):
        super().__init__()
        # first multi-agent attention and then multi-window attention
        self.layers = nn.ModuleList([])
        self.num_blocks = num_blocks

        for _ in range(num_blocks):
            att = (
                HGTCavAttention(
                    cav_att_config["dim"], heads=cav_att_config["heads"], dim_head=cav_att_config["dim_head"], dropout=cav_att_config["dropout"]
                )
                if cav_att_config["use_hetero"]
                else CavAttention(
                    cav_att_config["dim"], heads=cav_att_config["heads"], dim_head=cav_att_config["dim_head"], dropout=cav_att_config["dropout"]
                )
            )
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(cav_att_config["dim"], att),
                        PreNorm(
                            cav_att_config["dim"],
                            PyramidWindowAttention(
                                pwindow_config["dim"],
                                heads=pwindow_config["heads"],
                                dim_heads=pwindow_config["dim_head"],
                                drop_out=pwindow_config["dropout"],
                                window_size=pwindow_config["window_size"],
                                relative_pos_embedding=pwindow_config["relative_pos_embedding"],
                                fuse_method=pwindow_config["fusion_method"],
                            ),
                        ),
                    ]
                )
            )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        prior_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through V2X fusion block.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, L, H, W, C).
        mask : Tensor
            Attention mask for valid agents.
        prior_encoding : Tensor
            Prior encoding with temporal and infrastructure information.

        Returns
        -------
        Tensor
            Fused features with shape (B, L, H, W, C).
        """
        for cav_attn, pwindow_attn in self.layers:
            x = cav_attn(x, mask=mask, prior_encoding=prior_encoding) + x
            x = pwindow_attn(x) + x
        return x


class V2XTEncoder(nn.Module):
    """
    V2X Transformer Encoder module.
    
    Parameters
    ----------
    Dict[str, Any]
        Dictionary containing encoder configuration:
        
        - cav_att_config : dict
            Configuration for CAV attention.
        - pwindow_att_config : dict
            Configuration for pyramid window attention.
        - feed_forward : dict
            Feed-forward network configuration.
        - num_blocks : int
            Number of fusion blocks.
        - depth : int
            Number of encoder layers.
        - sttf : dict
            Spatial-temporal transformation config.
        - use_roi_mask : bool
            Whether to use ROI masking.
        
    Attributes
    ----------
    downsample_rate : int
        Feature downsampling rate.
    discrete_ratio : float
        Discretization ratio.
    use_roi_mask : bool
        Flag indicating whether ROI masking is used.
    use_RTE : bool
        Flag indicating whether relative temporal encoding is used.
    RTE_ratio : float
        Temporal encoding scaling ratio.
    sttf : STTF
        Spatial-temporal transformation fusion module.
    prior_feed : nn.Linear
        Linear layer to adjust channel numbers from C+3 to C.
    layers : nn.ModuleList
        List of encoder layers (fusion block + feed-forward).
    rte : RTE, optional
        Relative temporal encoding module if use_RTE is True.
    """
    
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()

        cav_att_config = args["cav_att_config"]
        pwindow_att_config = args["pwindow_att_config"]
        feed_config = args["feed_forward"]

        num_blocks = args["num_blocks"]
        depth = args["depth"]
        mlp_dim = feed_config["mlp_dim"]
        dropout = feed_config["dropout"]

        self.downsample_rate = args["sttf"]["downsample_rate"]
        self.discrete_ratio = args["sttf"]["voxel_size"][0]
        self.use_roi_mask = args["use_roi_mask"]
        self.use_RTE = cav_att_config["use_RTE"]
        self.RTE_ratio = cav_att_config["RTE_ratio"]
        self.sttf = STTF(args["sttf"])
        # adjust the channel numbers from 256+3 -> 256
        self.prior_feed = nn.Linear(cav_att_config["dim"] + 3, cav_att_config["dim"])
        self.layers = nn.ModuleList([])
        if self.use_RTE:
            self.rte = RTE(cav_att_config["dim"], self.RTE_ratio)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        V2XFusionBlock(num_blocks, cav_att_config, pwindow_att_config),
                        PreNorm(cav_att_config["dim"], FeedForward(cav_att_config["dim"], mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        spatial_correction_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through V2X encoder.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, L, H, W, C+3) where last 3 channels
            contain velocity, time_delay, and infrastructure information.
        mask : Tensor
            Mask indicating valid agents.
        spatial_correction_matrix : Tensor
            Spatial correction matrices for feature alignment.

        Returns
        -------
        Tensor
            Encoded features with shape (B, L, H, W, C).
        """
        # transform the features to the current timestamp
        # velocity, time_delay, infra
        # (B,L,H,W,3)
        prior_encoding = x[..., -3:]
        # (B,L,H,W,C)
        x = x[..., :-3]
        if self.use_RTE:
            # dt: (B,L)
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)
            x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        com_mask = (
            mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            if not self.use_roi_mask
            else get_roi_and_cav_mask(x.shape, mask, spatial_correction_matrix, self.discrete_ratio, self.downsample_rate)
        )
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x


class V2XTransformer(nn.Module):
    """
    V2X Transformer for multi-agent feature fusion.

    This is the main transformer architecture that combines spatial-temporal alignment,
    relative temporal encoding, and multi-scale attention for robust cooperative perception.

    Parameters
    ----------
    args : dict of str to Any
        Configuration dictionary containing:
        - 'encoder': Encoder configuration for V2XTEncoder.

    Attributes
    ----------
    encoder : V2XTEncoder
        Transformer encoder module.
    """

    def __init__(self, args: Dict[str, Any]):
        super(V2XTransformer, self).__init__()

        encoder_args = args["encoder"]
        self.encoder = V2XTEncoder(encoder_args)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        spatial_correction_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through V2X Transformer.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, L, H, W, C+3).
        mask : Tensor
            Mask indicating valid agents.
        spatial_correction_matrix : Tensor
            Spatial correction matrices for feature alignment.

        Returns
        -------
        Tensor
            Ego agent's fused features with shape (B, H, W, C).
        """
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output
