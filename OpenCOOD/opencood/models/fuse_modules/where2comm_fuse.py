"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention


class Communication(nn.Module):
    def __init__(self, args):
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

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def build_mask(self, confidence_map, *, force_first: bool):
        """
        Build the spatial transmission mask for one group of agents.

        Parameters
        ----------
        confidence_map : torch.Tensor
            Per-agent classification logits.
        force_first : bool
            Whether the first feature belongs to the local receiver and must
            remain unmasked.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Binary communication mask and selected-location ratio.
        """
        _, _, height, width = confidence_map.shape
        original_map, _ = confidence_map.sigmoid().max(dim=1, keepdim=True)
        communication_map = self.gaussian_filter(original_map) if self.smooth else original_map
        agent_count = communication_map.shape[0]

        if self.training:
            selected_count = int(height * width * random.uniform(0, 1))
            flattened_map = communication_map.reshape(agent_count, height * width)
            _, indices = torch.topk(flattened_map, k=selected_count, sorted=False)
            communication_mask = torch.zeros_like(flattened_map)
            selected_values = torch.ones(
                agent_count,
                selected_count,
                dtype=communication_map.dtype,
                device=communication_map.device,
            )
            communication_mask = torch.scatter(
                communication_mask,
                -1,
                indices,
                selected_values,
            ).reshape(agent_count, 1, height, width)
        elif self.threshold:
            communication_mask = torch.where(
                communication_map > self.threshold,
                torch.ones_like(communication_map),
                torch.zeros_like(communication_map),
            )
        else:
            communication_mask = torch.ones_like(communication_map)

        communication_rate = communication_mask.sum() / (agent_count * height * width)
        if force_first:
            communication_mask[0] = 1
        return communication_mask, communication_rate

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        communication_masks = []
        communication_rates = []
        for b in range(B):
            communication_mask, communication_rate = self.build_mask(
                batch_confidence_maps[b],
                force_first=True,
            )
            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Where2comm(nn.Module):
    def __init__(self, args):
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

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
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
