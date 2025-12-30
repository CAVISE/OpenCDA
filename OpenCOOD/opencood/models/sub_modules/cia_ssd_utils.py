"""
Utility functions and modules for CIA-SSD.
This module provides building blocks for the CIA-SSD detector, including feature fusion
and multi-scale feature extraction components.
"""
import torch
from torch import nn
from typing import Dict, List, Union

class SSFA(nn.Module):
    """
    Single-Stage Feature Aggregation (SSFA) module for feature extraction and fusion.
    
    This module processes input features through multiple convolutional blocks and fuses
    features at different scales using skip connections and attention mechanisms.
    Args:
        args (dict): Configuration dictionary containing:
            - feature_num (int): Number of input features/channels.
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
        Forward pass of the SSFA module.
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
    Build convolutional layers. kernel_size, stride and padding should be a list with the lengths that match n_layers
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
        Forward pass of the detection head.
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
