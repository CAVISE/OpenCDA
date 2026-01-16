import torch
import numpy as np

from einops import rearrange
from opencood.utils.common_utils import torch_tensor_to_numpy

from typing import List

def regroup(
    dense_feature: torch.Tensor,
    record_len: List[int],
    max_len: int
) -> torch.Tensor:
    """
    Regroup concatenated CAV features into batched format with padding.
    
    Converts variable-length concatenated features into fixed-size batch
    tensor by padding with zeros, enabling batched processing of multi-agent
    data with different numbers of CAVs per sample.

    Parameters
    ----------
    dense_feature : torch.Tensor
        Concatenated features from all CAVs with shape (N_total, C, H, W),
        where N_total = sum(record_len).
    record_len : list of int
        Number of CAVs in each sample, e.g., [3, 5, 2] for batch size 3.
    max_len : int
        Maximum number of CAVs to pad to (typically 5 for V2V scenarios).

    Returns
    -------
    regroup_features : torch.Tensor
        Regrouped and padded features with shape (B, max_len, C, H, W),
        where B is batch size. Padded positions contain zeros.
    mask : torch.Tensor
        Binary mask with shape (B, max_len) indicating valid CAVs.
        mask[i, j] = 1 if CAV j exists in sample i, else 0.

    """
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        # M, C, H, W
        feature_shape = split_feature.shape

        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(padding_len, feature_shape[1], feature_shape[2], feature_shape[3])
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor], dim=0)

        # 1, 5C, H, W
        split_feature = split_feature.view(-1, feature_shape[2], feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = rearrange(regroup_features, "b (l c) h w -> b l c h w", l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask
