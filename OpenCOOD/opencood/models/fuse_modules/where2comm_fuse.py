"""
Implementation of Where2comm fusion.

This module implements Where2comm, a communication-efficient multi-agent fusion
method that selectively shares features based on confidence maps.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention


class Communication(nn.Module):
    """
    Communication module for Where2comm that handles feature masking based on confidence.

    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing configuration:

        - threshold : float
            Confidence threshold for communication.
        - gaussian_smooth : dict, optional
            Dictionary with Gaussian smoothing parameters:

            - k_size : int
                Kernel size for Gaussian smoothing.
            - c_sigma : float
                Sigma value for Gaussian kernel.

    Attributes
    ----------
    threshold : float
        Confidence threshold for determining which features to communicate.
    smooth : bool
        Flag indicating whether Gaussian smoothing is applied.
    gaussian_filter : nn.Conv2d, optional
        Gaussian filter for smoothing confidence maps if smoothing is enabled.
    """

    def __init__(self, args: Dict[str, Any]):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args["threshold"]
        if "gaussian_smooth" in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args["gaussian_smooth"]["k_size"]
            c_sigma = args["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size: int = 5, sigma: float = 1.0) -> None:
        """
        Initialize Gaussian filter weights.

        Parameters
        ----------
        k_size : int, optional
            Kernel size for Gaussian filter.
        sigma : float, optional
            Standard deviation for Gaussian kernel.
        """
        center = k_size // 2
        x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps: List[torch.Tensor], B: int) -> Tuple[torch.Tensor, float]:
        """
        Generate communication masks based on confidence maps.

        Parameters
        ----------
        batch_confidence_maps : list of torch.Tensor
            List of confidence maps with shapes [(L1, H, W), (L2, H, W), ...].
        B : int
            Batch size.

        Returns
        -------
        communication_masks : torch.Tensor
            Binary masks for communication of shape (sum(L), 1, H, W).
        communication_rate : float
            Average communication rate across batch.
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion module using scaled dot-product attention.

    This module applies self-attention across spatial dimensions to fuse features
    from multiple agents at each pixel location.

    Parameters
    ----------
    feature_dim : int
        Dimension of input features

    Attributes
    ----------
    att : ScaledDotProductAttention
        Scaled dot-product attention module.
    """

    def __init__(self, feature_dim: int):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention fusion.

        Parameters
        ----------
        x : Tensor
            Input features with shape (cav_num, C, H, W).

        Returns
        -------
        Tensor
            Fused features for ego agent with shape (C, H, W).
        """
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Where2comm(nn.Module):
    """
    Where2comm fusion module for communication-efficient multi-agent perception.

    This module implements Where2comm fusion strategy that selectively communicates
    features based on confidence maps to reduce bandwidth while maintaining detection
    performance. Supports both single-scale and multi-scale fusion.

    Parameters
    ----------
    args : dict of str to Any
        Configuration dictionary containing:
        - 'voxel_size': Voxel size [x, y, z].
        - 'downsample_rate': Downsampling rate for features.
        - 'fully': Whether to use fully connected communication graph.
        - 'multi_scale': Whether to use multi-scale fusion.
        - 'layer_nums': Number of layers at each scale (if multi_scale is True).
        - 'num_filters': Number of filters at each scale (if multi_scale is True).
        - 'in_channels': Number of input channels (if multi_scale is False).
        - 'communication': Configuration for communication module.

    Attributes
    ----------
    discrete_ratio : float
        Discretization ratio from voxel size.
    downsample_rate : int
        Feature downsampling rate.
    fully : bool
        Flag indicating whether to use fully connected communication.
    multi_scale : bool
        Flag indicating whether to use multi-scale fusion.
    num_levels : int, optional
        Number of pyramid levels if multi_scale is True.
    fuse_modules : nn.ModuleList or AttentionFusion
        Fusion modules for each scale or single fusion module.
    naive_communication : Communication
        Communication module for generating masks.
    """

    def __init__(self, args: Dict[str, Any]):
        super(Where2comm, self).__init__()
        self.discrete_ratio = args["voxel_size"][0]
        self.downsample_rate = args["downsample_rate"]

        self.fully = args["fully"]
        if self.fully:
            print("constructing a fully connected communication graph")
        else:
            print("constructing a partially connected communication graph")

        self.multi_scale = args["multi_scale"]
        if self.multi_scale:
            layer_nums = args["layer_nums"]
            num_filters = args["num_filters"]
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args["in_channels"])

        self.naive_communication = Communication(args["communication"])

    def regroup(x: torch.Tensor, record_len: torch.Tensor) -> List[torch.Tensor]:
        """
        Regroup features based on record lengths.

        Parameters
        ----------
        x : Tensor
            Features with shape (sum(n_cav), C, H, W).
        record_len : Tensor
            Number of agents per batch sample with shape (B,).

        Returns
        -------
        list of Tensor
            List of feature tensors [(L1, C, H, W), (L2, C, H, W), ...].
        """
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(
        self,
        x: torch.Tensor,
        psm_single: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        backbone: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Where2comm fusion.

        Parameters
        ----------
        x : Tensor
            Input features with shape (sum(n_cav), C, H, W).
        psm_single : Tensor
            Single-agent confidence maps for communication masking.
        record_len : Tensor
            Number of agents per batch sample with shape (B,).
        pairwise_t_matrix : Tensor
            Pairwise transformation matrices with shape (B, L, L, 4, 4).
        backbone : nn.Module, optional
            Backbone network for multi-scale processing. Required if multi_scale is True.

        Returns
        -------
        tuple of (Tensor, Tensor)
            - x_fuse: Fused features with shape (B, C, H, W).
            - communication_rates: Communication rate indicating bandwidth usage.
        """

        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]

        if self.multi_scale:
            ups = []

            for i in range(self.num_levels):
                x = backbone.blocks[i](x)

                # 1. Communication (mask the features)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Prune
                        batch_confidence_maps = self.regroup(psm_single, record_len)
                        communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                        if x.shape[-1] != communication_masks.shape[-1]:
                            communication_masks = F.interpolate(
                                communication_masks, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False
                            )
                        x = x * communication_masks

                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            # 1. Communication (mask the features)
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                # Prune
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                x = x * communication_masks

            # 2. Split the features
            # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
            # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        return x_fuse, communication_rates
