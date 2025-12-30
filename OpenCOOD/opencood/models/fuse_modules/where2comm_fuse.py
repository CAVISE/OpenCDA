"""
Implementation of Where2comm fusion.
"""
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor 
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
        center = k_size // 2
        x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(
        self, 
        batch_confidence_maps: List[Tensor], 
        B: int
    ) -> Tuple[Tensor, float]:
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
    Attention-based feature fusion module.
    
    Parameters
    ----------
    feature_dim : int
        Dimension of input features.
    """
    
    def __init__(self, feature_dim: int) -> None:
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Where2comm(nn.Module):
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

    def regroup(x: Tensor, record_len: Tensor) -> List[Tensor]:
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(
        self, 
        x: Tensor, 
        psm_single: Tensor, 
        record_len: Tensor, 
        pairwise_t_matrix: Tensor, 
        backbone: Optional[nn.Module] = None
    ) -> Tensor:
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (sum(n_cav), C, H, W).
        psm_single : torch.Tensor
            Single probability score map.
        record_len : torch.Tensor
            List of shape (B,) indicating number of CAVs per sample.
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each CAV to ego of shape (B, L, L, 4, 4).
        backbone : nn.Module, optional
            Backbone network. Default is None.
        
        Returns
        -------
        torch.Tensor
            Fused feature.
        """
