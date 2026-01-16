"""
Inference utility functions for different fusion strategies in cooperative perception.

This module provides inference functions for late fusion, early fusion, and
intermediate fusion strategies, along with utilities for saving predictions
and ground truth data.
"""

import os
from collections import OrderedDict
from typing import Any, Tuple

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data: dict, model: torch.nn.Module, dataset: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
        Dictionary containing data for each CAV (Connected Autonomous Vehicle).
    model : torch.nn.Module
        The trained model for inference.
    dataset : Any
        Dataset object with post_process method (e.g., LateFusionDataset).

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    pred_score : torch.Tensor
        The confidence scores for predictions.
    gt_box_tensor : torch.Tensor
        The tensor of ground truth bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data: dict, model: torch.nn.Module, dataset: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
        Dictionary containing data for the ego vehicle.
    model : torch.nn.Module
        The trained model for inference.
    dataset : Any
        Dataset object with post_process method (e.g., EarlyFusionDataset).

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    pred_score : torch.Tensor
        The confidence scores for predictions.
    gt_box_tensor : torch.Tensor
        The tensor of ground truth bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]

    output_dict["ego"] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data: dict, model: torch.nn.Module, dataset: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Model inference for intermediate fusion.

    Parameters
    ----------
    batch_data : dict
        Dictionary containing data for the ego vehicle.
    model : torch.nn.Module
        The trained model for inference.
    dataset : Any
        Dataset object with post_process method (e.g., IntermediateFusionDataset).

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    pred_score : torch.Tensor
        The confidence scores for predictions.
    gt_box_tensor : torch.Tensor
        The tensor of ground truth bounding box.

    Notes
    -----
    This function currently uses the same implementation as early fusion.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, pcd: torch.Tensor, timestamp: int, save_path: str) -> None:
    """
    Save prediction and ground truth tensors to numpy files.

    Parameters
    ----------
    pred_tensor : torch.Tensor or npt.NDArray
        Predicted bounding boxes.
    gt_tensor : torch.Tensor or npt.NDArray
        Ground truth bounding boxes.
    pcd : torch.Tensor or npt.NDArray
        Point cloud data.
    timestamp : int
        Timestamp or frame number for filename.
    save_path : str
        Directory path where files will be saved.

    Notes
    -----
    Saves three files:
        - {timestamp:04d}_pcd.npy: Point cloud data
        - {timestamp:04d}_pred.npy: Predicted bounding boxes
        - {timestamp:04d}_gt.npy_test: Ground truth bounding boxes
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, "%04d_pcd.npy" % timestamp), pcd_np)
    np.save(os.path.join(save_path, "%04d_pred.npy" % timestamp), pred_np)
    np.save(os.path.join(save_path, "%04d_gt.npy_test" % timestamp), gt_np)
