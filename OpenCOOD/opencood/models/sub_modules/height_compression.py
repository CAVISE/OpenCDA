"""
Height Compression Module for Sparse 3D Features.

This module compresses 3D sparse features along the height dimension
to create a 2D Bird's Eye View (BEV) representation.
"""

from typing import Dict, Any
from torch import nn

class HeightCompression(nn.Module):
    """
    Height compression for converting 3D features to 2D BEV.

    This module flattens the height dimension of 3D sparse features
    into the channel dimension, creating a dense 2D BEV feature map
    suitable for downstream 2D convolutions.

    Parameters
    ----------
    model_cfg : dict of str to Any
        Model configuration dictionary containing:
        - 'feature_num': Number of output BEV feature channels.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    num_bev_features : int
        Number of output BEV feature channels.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg["feature_num"]

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for processing batch data.
        Parameters
        ----------
        batch_dict : dict of str to Any
            Batch dictionary containing:
            - 'encoded_spconv_tensor': Sparse 3D features with spatial
              shape (D, H, W) and C channels.
            - 'encoded_spconv_tensor_stride': Downsampling stride.

        Returns
        -------
        dict of str to Any
            Updated batch dictionary with:
            - 'spatial_features': Dense 2D BEV features with shape (N, C*D, H, W).
            - 'spatial_features_stride': Downsampling stride.
        """
        encoded_spconv_tensor = batch_dict["encoded_spconv_tensor"]
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict["spatial_features"] = spatial_features
        batch_dict["spatial_features_stride"] = batch_dict["encoded_spconv_tensor_stride"]
        return batch_dict
