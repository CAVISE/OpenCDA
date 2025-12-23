from typing import Dict, Any
import torch
from torch import nn, Tensor

class HeightCompression(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], **kwargs) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg["feature_num"]

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict["encoded_spconv_tensor"]
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict["spatial_features"] = spatial_features
        batch_dict["spatial_features_stride"] = batch_dict["encoded_spconv_tensor_stride"]
        return batch_dict
