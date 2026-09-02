"""Tests for GT-independent FPV-RCNN RoI inference."""

from unittest.mock import MagicMock

import torch
from torch import nn

from opencood.models.sub_modules.roi_head import RoIHead


def _build_minimal_roi_head(proposal_count: int) -> RoIHead:
    head = RoIHead.__new__(RoIHead)
    nn.Module.__init__(head)
    head.code_size = 7
    head.grid_size = 1
    head.roi_grid_pool = MagicMock(return_value=torch.ones((proposal_count, 1, 1), dtype=torch.float32))
    head.shared_fc_layers = nn.Identity()
    head.cls_layers = nn.Identity()
    head.iou_layers = nn.Identity()
    head.reg_layers = nn.Identity()
    return head


def test_roi_head_inference_prepares_rois_without_ground_truth_assignment():
    boxes_by_scene = [
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25],
                [7.0, 8.0, 9.0, 2.0, 3.0, 4.0, -0.5],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 1.25]],
            dtype=torch.float32,
        ),
    ]
    head = _build_minimal_roi_head(proposal_count=3)
    head.assign_targets = MagicMock(side_effect=AssertionError("assign_targets must not run during inference"))
    head.eval()
    batch = {
        "boxes_fused": boxes_by_scene,
        "record_len": torch.tensor([1, 1]),
    }

    output = head(batch)

    head.assign_targets.assert_not_called()
    assert output["rcnn_label_dict"]["record_len"] == [2, 1]
    assert output["rcnn_label_dict"]["rois"].shape == (3, 7)
    anchors = output["rcnn_label_dict"]["rois_anchor"]
    torch.testing.assert_close(anchors[:, :3], torch.zeros((3, 3)))
    torch.testing.assert_close(anchors[:, 6], torch.zeros(3))
    torch.testing.assert_close(anchors[:, 3:6], output["rcnn_label_dict"]["rois"][:, 3:6])
    assert set(output["fpvrcnn_out"]) == {"rcnn_cls", "rcnn_iou", "rcnn_reg"}


def test_roi_head_training_keeps_target_assignment_path():
    head = _build_minimal_roi_head(proposal_count=1)
    head.assign_targets = MagicMock(
        side_effect=lambda batch: {
            **batch,
            "rcnn_label_dict": {
                "rois": torch.ones((1, 7), dtype=torch.float32),
                "rois_anchor": torch.ones((1, 7), dtype=torch.float32),
                "record_len": [1],
            },
        }
    )
    head.prepare_rois = MagicMock(side_effect=AssertionError("prepare_rois must not replace training targets"))
    head.train()

    head({"record_len": torch.tensor([1])})

    head.assign_targets.assert_called_once()
    head.prepare_rois.assert_not_called()
