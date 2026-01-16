"""
Utility Functions and Modules for CIA-SSD.

This module provides building blocks for the CIA-SSD detector, including
Single-Stage Feature Aggregation (SSFA) and detection head components for
multi-scale feature extraction and fusion.
"""

import torch
from torch import nn
from typing import Dict, List, Union

class SSFA(nn.Module):
    """
    Single-Stage Feature Aggregation module.

    This module processes input features through multiple convolutional blocks
    and fuses features at different scales using skip connections and attention
    mechanisms for enhanced feature representation.

    Parameters
    ----------
    args : dict
        Configuration dictionary containing:
        - 'feature_num': Number of input feature channels (int).

    Attributes
    ----------
    _num_input_features : int
        Number of input feature channels (128).
    bottom_up_block_0 : nn.Sequential
        First bottom-up convolution block.
    bottom_up_block_1 : nn.Sequential
        Second bottom-up convolution block with downsampling.
    trans_0 : nn.Sequential
        Transition layer for scale 0 features.
    trans_1 : nn.Sequential
        Transition layer for scale 1 features.
    deconv_block_0 : nn.Sequential
        Deconvolution block for upsampling scale 1 to scale 0.
    deconv_block_1 : nn.Sequential
        Deconvolution block for upsampling scale 1.
    conv_0 : nn.Sequential
        Convolution layer for refining scale 0 features.
    conv_1 : nn.Sequential
        Convolution layer for refining scale 1 features.
    w_0 : nn.Sequential
        Weight prediction layer for scale 0.
    w_1 : nn.Sequential
        Weight prediction layer for scale 1.
    """
    
    def __init__(self, args: Dict):
        super(SSFA, self).__init__()
        self._num_input_features = args["feature_num"]  # 128

        seq = [nn.ZeroPad2d(1)] + get_conv_layers(
            "Conv2d", 128, 128, n_layers=3, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1], sequential=False
        )
        self.bottom_up_block_0 = nn.Sequential(*seq)
        self.bottom_up_block_1 = get_conv_layers("Conv2d", 128, 256, n_layers=3, kernel_size=[3, 3, 3], stride=[2, 1, 1], padding=[1, 1, 1])

        self.trans_0 = get_conv_layers("Conv2d", 128, 128, n_layers=1, kernel_size=[1], stride=[1], padding=[0])
        self.trans_1 = get_conv_layers("Conv2d", 256, 256, n_layers=1, kernel_size=[1], stride=[1], padding=[0])

        self.deconv_block_0 = get_conv_layers("ConvTranspose2d", 256, 128, n_layers=1, kernel_size=[3], stride=[2], padding=[1], output_padding=[1])
        self.deconv_block_1 = get_conv_layers("ConvTranspose2d", 256, 128, n_layers=1, kernel_size=[3], stride=[2], padding=[1], output_padding=[1])

        self.conv_0 = get_conv_layers("Conv2d", 128, 128, n_layers=1, kernel_size=[3], stride=[1], padding=[1])
        self.conv_1 = get_conv_layers("Conv2d", 128, 128, n_layers=1, kernel_size=[3], stride=[1], padding=[1])

        self.w_0 = get_conv_layers("Conv2d", 128, 1, n_layers=1, kernel_size=[1], stride=[1], padding=[0], relu_last=False)
        self.w_1 = get_conv_layers("Conv2d", 128, 1, n_layers=1, kernel_size=[1], stride=[1], padding=[0], relu_last=False)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SSFA module.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Fused output features with shape (B, 128, H, W).
        """
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]

        return x_output.contiguous()


def get_conv_layers(conv_name: str, 
                   in_channels: int, 
                   out_channels: int, 
                   n_layers: int, 
                   kernel_size: List[int], 
                   stride: List[int], 
                   padding: List[int], 
                   relu_last: bool = True, 
                   sequential: bool = True, 
                   **kwargs) -> Union[nn.Sequential, List[nn.Module]]:
    """
    Build convolutional layers with batch normalization and ReLU.

    Parameters
    ----------
    conv_name : str
        Name of the convolution layer class (e.g., 'Conv2d', 'ConvTranspose2d').
    in_channels : int
        Number of input channels for first layer.
    out_channels : int
        Number of output channels for all layers.
    n_layers : int
        Number of convolutional layers to create.
    kernel_size : list of int
        Kernel sizes for each layer (length must match n_layers).
    stride : list of int
        Stride values for each layer (length must match n_layers).
    padding : list of int
        Padding values for each layer (length must match n_layers).
    relu_last : bool, optional
        Whether to apply ReLU after the last layer. Default is True.
    sequential : bool, optional
        If True, return nn.Sequential. If False, return list of modules.
        Default is True.
    **kwargs
        Additional keyword arguments passed to convolution layers. Each value
        should be a list with length matching n_layers.

    Returns
    -------
    nn.Sequential or list of nn.Module
        Sequential container or list of modules (Conv-BN-ReLU blocks).
    """
    seq = []
    for i in range(n_layers):
        seq.extend(
            [
                getattr(nn, conv_name)(
                    in_channels,
                    out_channels,
                    kernel_size[i],
                    stride=stride[i],
                    padding=padding[i],
                    bias=False,
                    **{k: v[i] for k, v in kwargs.items()},
                ),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            ]
        )
        if i < n_layers - 1 or relu_last:
            seq.append(nn.ReLU())
        in_channels = out_channels
    if sequential:
        return nn.Sequential(*seq)
    else:
        return seq


class Head(nn.Module):
    """
    Detection head for predicting bounding boxes, classes, and IoU scores.

    This module implements the final prediction layers for object detection,
    outputting box coordinates, class probabilities, IoU scores, and optionally
    direction classifications.

    Parameters
    ----------
    num_input : int
        Number of input feature channels.
    num_pred : int
        Number of box prediction parameters (e.g., 14 for 7-DoF boxes with 2 anchors).
    num_cls : int
        Number of class prediction outputs.
    num_iou : int, optional
        Number of IoU prediction outputs. Default is 2.
    use_dir : bool, optional
        Whether to predict direction classification. Default is False.
    num_dir : int, optional
        Number of direction classification outputs. Default is 1.

    Attributes
    ----------
    use_dir : bool
        Whether direction prediction is enabled.
    conv_box : nn.Conv2d
        Convolution layer for box prediction.
    conv_cls : nn.Conv2d
        Convolution layer for class prediction.
    conv_iou : nn.Conv2d
        Convolution layer for IoU prediction.
    conv_dir : nn.Conv2d, optional
        Convolution layer for direction prediction (if use_dir=True).
    """

    def __init__(self, 
                num_input: int, 
                num_pred: int, 
                num_cls: int, 
                num_iou: int = 2, 
                use_dir: bool = False, 
                num_dir: int = 1):
        super(Head, self).__init__()
        self.use_dir = use_dir

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)  # 128 -> 14
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)  # 128 -> 2
        self.conv_iou = nn.Conv2d(num_input, num_iou, 1, bias=False)

        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)  # 128 -> 4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through detection head.

        Parameters
        ----------
        x : Tensor
            Input features with shape (B, C, H, W).

        Returns
        -------
        dict of str to Tensor
            Dictionary containing:
            - 'box_preds': Box predictions with shape (B, num_pred, H, W).
            - 'cls_preds': Class predictions with shape (B, num_cls, H, W).
            - 'iou_preds': IoU predictions with shape (B, num_iou, H, W).
            - 'dir_cls_preds': Direction predictions with shape (B, num_dir, H, W)
              if use_dir=True, otherwise zeros with shape (B, 1, 2).
        """
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x)  # dir_preds.shape=[8, w, h, 4]
            ret_dict["dir_cls_preds"] = dir_preds
        else:
            ret_dict["dir_cls_preds"] = torch.zeros((len(box_preds), 1, 2))

        ret_dict["iou_preds"] = self.conv_iou(x)

        return ret_dict
