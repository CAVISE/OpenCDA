import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class PixorLoss(nn.Module):
    """
    PIXOR loss function for 3D object detection.
    
    This loss combines classification and regression losses for the PIXOR model,
    which predicts objects in a bird's eye view representation.
    """
    def __init__(self, args: Dict[str, float]) -> None:
        super(PixorLoss, self).__init__()
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.loss_dict = {}

    def dtype(self) -> torch.dtype:
        """
        Get the default data type for the loss computation.
        
        Returns:
            torch.dtype: The default data type (float16)
        """
        return torch.float16

    def forward(self, output_dict: Dict[str, torch.Tensor], 
               target_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for pixor network.
        
        Args:
            output_dict: Dictionary containing model outputs:
                - cls: Classification predictions [B, 1, H, W]
                - reg: Regression predictions [B, 6, H, W]
            target_dict: Dictionary containing ground truth:
                - label_map: Target tensor [B, 7, H, W] where channels are:
                    - channel 0: classification target
                    - channels 1-6: regression targets
        Returns:
            torch.Tensor: Total loss value
        """
        targets = target_dict["label_map"]
        cls_preds, loc_preds = output_dict["cls"], output_dict["reg"]

        cls_targets, loc_targets = targets.split([1, 6], dim=1)
        pos_count = cls_targets.sum()
        neg_count = (cls_targets == 0).sum()

        w1 = (neg_count / (pos_count + neg_count)).to(dtype=self.dtype())
        w2 = (pos_count / (pos_count + neg_count)).to(dtype=self.dtype())

        weights = torch.ones_like(cls_preds.reshape(-1), dtype=self.dtype(), device=cls_preds.device)
        weights[cls_targets.reshape(-1) == 1] = w1
        weights[cls_targets.reshape(-1) == 0] = w2

        # cls_loss = F.binary_cross_entropy_with_logits(input=cls_preds.reshape(-1), target=cls_targets.reshape(-1), weight=weights,
        #                                               reduction='mean')

        print(f"cls_preds shape: {cls_preds.shape}")
        print(f"cls_targets shape: {cls_targets.shape}")
        # print(f"pos_neg_weights shape: {pos_neg_weights.shape}")

        if cls_targets.shape[2:] != cls_preds.shape[2:]:
            cls_targets = F.interpolate(cls_targets, size=cls_preds.shape[2:], mode="bilinear", align_corners=False)
            # pos_neg_weights = F.interpolate(pos_neg_weights, size=cls_preds.shape[2:], mode='bilinear', align_corners=False)
            print(f"Resized cls_targets shape: {cls_targets.shape}")
        # print(f"Resized pos_neg_weights shape: {pos_neg_weights.shape}")

        cls_loss = F.binary_cross_entropy_with_logits(input=cls_preds, target=cls_targets, reduction="mean")
        pos_pixels = cls_targets.sum()

        loc_loss = F.smooth_l1_loss(cls_targets * loc_preds, cls_targets * loc_targets, reduction="sum")
        loc_loss = loc_loss / pos_pixels if pos_pixels > 0 else loc_loss

        total_loss = self.alpha * cls_loss + self.beta * loc_loss

        self.loss_dict.update({"total_loss": total_loss, "reg_loss": loc_loss, "cls_loss": cls_loss})

        return total_loss

    def logging(self, epoch: int, batch_id: int, batch_len: int, writer, pbar: Optional[Any] = None) -> None:
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict["total_loss"]
        reg_loss = self.loss_dict["reg_loss"]
        cls_loss = self.loss_dict["cls_loss"]

        if pbar is None:
            print(
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (epoch, batch_id + 1, batch_len, total_loss.item(), cls_loss.item(), reg_loss.item())
            )
        else:
            pbar.set_description(
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (epoch, batch_id + 1, batch_len, total_loss.item(), cls_loss.item(), reg_loss.item())
            )

        writer.add_scalar("Regression_loss", reg_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Confidence_loss", cls_loss.item(), epoch * batch_len + batch_id)


def test():
    torch.manual_seed(0)
    loss = PixorLoss(None)
    pred = torch.sigmoid(torch.randn(1, 7, 2, 3))
    label = torch.zeros(1, 7, 2, 3)
    loss = loss(pred, label)
    print(loss)


if __name__ == "__main__":
    test()
