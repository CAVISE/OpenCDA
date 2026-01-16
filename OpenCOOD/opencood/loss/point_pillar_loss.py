"""
Loss functions for PointPillar 3D object detection.

This module implements focal loss for classification and weighted smooth L1 loss
for bounding box regression in the PointPillar architecture. 
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
    
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    
    where x = input - target.
    
    Parameters
    ----------
    beta : float, optional
        Scalar float. L1 to L2 change point.
        For beta values < 1e-5, L1 loss is computed.
        Default is 1.0 / 9.0.
    code_weights : list of float, optional
        Code-wise weights. If None, no weights are applied.
        Default is None.

    Attributes
    ----------
    beta : float
        Stored transition threshold.
    code_weights : torch.Tensor or None
        Per-dimension weight tensor on CUDA device.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: Optional[List[float]] = None):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute the smooth L1 loss of differences.
        
        Parameters
        ----------
        diff : torch.Tensor
            The difference between predictions and targets.
        beta : float
            The L1 to L2 change point.
        
        Returns
        -------
        torch.Tensor
            The computed smooth L1 loss.
        """
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for weighted smooth L1 loss computation.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values with shape (B, N, C) where:
            - B: batch size
            - N: number of anchors/boxes
            - C: number of code dimensions
        target : torch.Tensor
            Ground truth values with shape (B, N, C).
            NaN values are ignored (replaced with predictions).
        weights : torch.Tensor or None, optional
            Anchor-wise weights with shape (B, N). Default is None.

        Returns
        -------
        loss : torch.Tensor
            Weighted smooth L1 loss with shape (B, N, C).
            If weights provided, each anchor's loss is scaled accordingly.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLoss(nn.Module):
    """
    Loss function for PointPillar object detection model.

    This loss combines focal loss for classification and weighted smooth L1 loss
    for bounding box regression, with sine encoding for rotation angles.

    Parameters
    ----------
    args : dict of str to Any
        Configuration dictionary containing:
        - 'cls_weight': Weight for classification loss.
        - 'reg': Weight coefficient for regression loss.

    Attributes
    ----------
    reg_loss_func : WeightedSmoothL1Loss
        Weighted smooth L1 loss for bounding box regression.
    alpha : float
        Alpha parameter for focal loss. Default is 0.25.
    gamma : float
        Gamma parameter for focal loss. Default is 2.0.
    cls_weight : float
        Weight for classification loss.
    reg_coe : float
        Weight coefficient for regression loss.
    loss_dict : dict
        Dictionary storing individual loss components.
    """

    def __init__(self, args: Dict[str, Any]):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args["cls_weight"]
        self.reg_coe = args["reg"]
        self.loss_dict = {}

    def forward(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the total loss for PointPillar model.

        Parameters
        ----------
        output_dict : dict of str to Tensor
            Model outputs containing:
            - 'rm': Regression map with shape (B, 7, H, W).
            - 'psm': Probability score map with shape (B, 1, H, W).
        target_dict : dict of str to Tensor
            Ground truth targets containing:
            - 'targets': Target bounding boxes with shape (B, N, 7).
            - 'pos_equal_one': Positive anchor labels with shape (B, H, W).

        Returns
        -------
        Tensor
            Total loss (scalar).
        """
        rm = output_dict["rm"]
        psm = output_dict["psm"]
        targets = target_dict["targets"]

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = target_dict["pos_equal_one"]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), 2, dtype=cls_preds.dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = reg_loss + conf_loss

        self.loss_dict.update({"total_loss": total_loss, "reg_loss": reg_loss, "conf_loss": conf_loss})

        return total_loss

    def cls_loss_func(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        Parameters
        ----------
        input : torch.Tensor
            Shape (B, #anchors, #classes). Predicted logits for each class.
        target : torch.Tensor
            Shape (B, #anchors, #classes). One-hot encoded classification targets.
        weights : torch.Tensor
            Shape (B, #anchors). Anchor-wise weights.
        
        Returns
        -------
        torch.Tensor
            Shape (B, #anchors, #classes). Weighted focal loss after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits.
        
        Computes: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        Reference: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        
        Parameters
        ----------
        input : torch.Tensor
            Shape (B, #anchors, #classes). Predicted logits for each class.
        target : torch.Tensor
            Shape (B, #anchors, #classes). One-hot encoded classification targets.
        
        Returns
        -------
        torch.Tensor
            Shape (B, #anchors, #classes). Sigmoid cross entropy loss without reduction.
        """
        loss = torch.clamp(input, min=0) - input * target + torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1: torch.Tensor, boxes2: torch.Tensor, dim: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add sine difference encoding for rotation angles.
        
        Converts rotation angle difference to sine representation for better
        gradient properties during training.
        
        Parameters
        ----------
        boxes1 : torch.Tensor
            Predicted bounding boxes with rotation angles.
        boxes2 : torch.Tensor
            Target bounding boxes with rotation angles.
        dim : int, optional
            Dimension index of the rotation angle. Default is 6.
        
        Returns
        -------
        boxes1_encoded : torch.Tensor
            Encoded predicted boxes with sine representation.
        boxes2_encoded : torch.Tensor
            Encoded target boxes with sine representation.
        """
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(boxes2[..., dim : dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(boxes2[..., dim : dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1)
        return boxes1, boxes2

    def logging(self, epoch: int, batch_id: int, batch_len: int, writer: Any, pbar: Optional[Any] = None) -> None:
        """
        Print out the loss function for current iteration.
        
        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch index.
        batch_len : int
            Total batch length in one iteration of training.
        writer : SummaryWriter
            TensorBoard SummaryWriter instance for visualization.
        pbar : tqdm, optional
            Progress bar instance. If None, prints to console instead.
            Default is None.
        """
        total_loss = self.loss_dict["total_loss"]
        reg_loss = self.loss_dict["reg_loss"]
        conf_loss = self.loss_dict["conf_loss"]

        if pbar is None:
            print(
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (epoch, batch_id + 1, batch_len, total_loss.item(), conf_loss.item(), reg_loss.item())
            )
        else:
            pbar.set_description(
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (epoch, batch_id + 1, batch_len, total_loss.item(), conf_loss.item(), reg_loss.item())
            )

        writer.add_scalar("Regression_loss", reg_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Confidence_loss", conf_loss.item(), epoch * batch_len + batch_id)
