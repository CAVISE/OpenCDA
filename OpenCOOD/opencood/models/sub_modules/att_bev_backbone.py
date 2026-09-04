import numpy as np
import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


class AttBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if "compression" in model_cfg and model_cfg["compression"] > 0:
            self.compress = True
            self.compress_layer = model_cfg["compression"]

        if "layer_nums" in self.model_cfg:
            assert len(self.model_cfg["layer_nums"]) == len(self.model_cfg["layer_strides"]) == len(self.model_cfg["num_filters"])

            layer_nums = self.model_cfg["layer_nums"]
            layer_strides = self.model_cfg["layer_strides"]
            num_filters = self.model_cfg["num_filters"]
        else:
            layer_nums = layer_strides = num_filters = []

        if "upsample_strides" in self.model_cfg:
            assert len(self.model_cfg["upsample_strides"]) == len(self.model_cfg["num_upsample_filter"])

            num_upsample_filters = self.model_cfg["num_upsample_filter"]
            upsample_strides = self.model_cfg["upsample_strides"]

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx], self.compress_layer - idx))

            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(int)
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict["spatial_features"]
        record_len = data_dict["record_len"]

        upsampled_features = []
        features = spatial_features
        for level, block in enumerate(self.blocks):
            features = block(features)
            if self.compress and level < len(self.compression_modules):
                features = self.compression_modules[level](features)
            fused_features = self.fuse_modules[level](features, record_len)
            if len(self.deblocks) > 0:
                upsampled_features.append(self.deblocks[level](fused_features))
            else:
                upsampled_features.append(fused_features)

        if len(upsampled_features) > 1:
            features = torch.cat(upsampled_features, dim=1)
        elif len(upsampled_features) == 1:
            features = upsampled_features[0]
        else:
            raise ValueError("At least one backbone level is required")

        if len(self.deblocks) > len(self.blocks):
            features = self.deblocks[-1](features)
        data_dict["spatial_features_2d"] = features
        return data_dict

    def encode_agent(self, spatial_features):
        """
        Compute every private backbone scale before feature fusion.

        Parameters
        ----------
        spatial_features : torch.Tensor
            Unfused BEV features for one or more agents.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Ordered private feature maps consumed by attention fusion.
        """
        feature_maps = []
        features = spatial_features
        for level, block in enumerate(self.blocks):
            features = block(features)
            if self.compress and level < len(self.compression_modules):
                encoded_features = self.compression_modules[level].encode(features)
                feature_maps.append(encoded_features)
                features = self.compression_modules[level].decode(encoded_features)
            else:
                feature_maps.append(features)
        return tuple(feature_maps)

    def fuse_agents(self, feature_maps, record_len):
        """
        Fuse encoded feature scales and decode the final BEV representation.

        Parameters
        ----------
        feature_maps : tuple[torch.Tensor, ...]
            Concatenated per-agent feature maps at every scale.
        record_len : torch.Tensor
            Number of successfully available agents in each sample.

        Returns
        -------
        torch.Tensor
            Decoded fused BEV feature map.
        """
        if len(feature_maps) != len(self.fuse_modules):
            raise ValueError("Unexpected number of intermediate feature scales")

        ups = []
        for level, features in enumerate(feature_maps):
            if self.compress and level < len(self.compression_modules):
                features = self.compression_modules[level].decode(features)
            fused_features = self.fuse_modules[level](features, record_len)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[level](fused_features))
            else:
                ups.append(fused_features)

        if len(ups) > 1:
            features = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            features = ups[0]
        else:
            raise ValueError("At least one intermediate feature scale is required")

        if len(self.deblocks) > len(self.blocks):
            features = self.deblocks[-1](features)
        return features
