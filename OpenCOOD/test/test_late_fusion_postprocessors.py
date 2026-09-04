"""Tests for sender-side late-fusion prediction decoding."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

mocked_opencood = sys.modules.get("opencood")
if mocked_opencood is not None and not hasattr(mocked_opencood, "__path__"):
    pytest.skip(
        "full OpenCOOD tests are disabled when the lightweight OpenCDA test doubles are active",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")

from opencood.data_utils.post_processor import ciassd_postprocessor  # noqa: E402
from opencood.data_utils.post_processor import voxel_postprocessor  # noqa: E402
from opencood.data_utils.post_processor.bev_postprocessor import (  # noqa: E402
    BevPostprocessor,
)
from opencood.data_utils.post_processor.ciassd_postprocessor import (  # noqa: E402
    CiassdPostprocessor,
)
from opencood.data_utils.post_processor.voxel_postprocessor import (  # noqa: E402
    VoxelPostprocessor,
)


def test_voxel_sender_decodes_candidates_without_nms_or_projection(monkeypatch):
    postprocessor = VoxelPostprocessor.__new__(VoxelPostprocessor)
    postprocessor.params = {
        "target_args": {"score_threshold": 0.5},
        "order": "hwl",
    }
    decoded_boxes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]]],
        dtype=torch.float32,
    )
    expected_corners = torch.ones((1, 8, 3), dtype=torch.float32)
    postprocessor.delta_to_boxes3d = MagicMock(return_value=decoded_boxes)
    corners = MagicMock(return_value=expected_corners)
    monkeypatch.setattr(voxel_postprocessor.box_utils, "boxes_to_corners_3d", corners)
    cav_content = {"anchor_box": torch.ones((1, 7), dtype=torch.float32)}
    output = {
        "psm": torch.tensor([[[[10.0]]]], dtype=torch.float32),
        "rm": torch.zeros((1, 7, 1, 1), dtype=torch.float32),
    }

    boxes, scores = postprocessor.decode_agent_predictions(cav_content, output)

    assert boxes is expected_corners
    assert scores.item() == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item())
    postprocessor.delta_to_boxes3d.assert_called_once_with(
        output["rm"],
        cav_content["anchor_box"],
    )
    corners.assert_called_once_with(decoded_boxes[0], order="hwl")


def test_bev_sender_decodes_only_above_threshold_candidates():
    postprocessor = BevPostprocessor.__new__(BevPostprocessor)
    postprocessor.params = {"target_args": {"score_threshold": 0.5}}
    postprocessor.denormalize_reg_map = MagicMock(side_effect=lambda value: value)
    expected_corners = torch.ones((1, 4, 2), dtype=torch.float32)
    postprocessor.reg_map_to_bbx_corners = MagicMock(return_value=expected_corners)
    output = {
        "cls": torch.tensor([[[[10.0, -10.0]]]], dtype=torch.float32),
        "reg": torch.zeros((1, 6, 1, 2), dtype=torch.float32),
    }

    boxes, scores = postprocessor.decode_agent_predictions({}, output)

    assert boxes is expected_corners
    assert scores.shape == (1,)
    assert scores.item() == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item())
    mask = postprocessor.reg_map_to_bbx_corners.call_args.args[1]
    torch.testing.assert_close(mask, torch.tensor([[True, False]]))


def test_ciassd_sender_applies_iou_and_direction_before_serialization(
    monkeypatch,
):
    postprocessor = CiassdPostprocessor.__new__(CiassdPostprocessor)
    postprocessor.params = {
        "target_args": {"score_threshold": 0.5},
        "order": "hwl",
    }
    decoded_boxes = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25],
                [7.0, 8.0, 9.0, 4.0, 5.0, 6.0, -0.5],
            ]
        ],
        dtype=torch.float32,
    )
    expected_corners = torch.ones((1, 8, 3), dtype=torch.float32)
    postprocessor.delta_to_boxes3d = MagicMock(return_value=decoded_boxes)
    corners = MagicMock(return_value=expected_corners)
    monkeypatch.setattr(ciassd_postprocessor.box_utils, "boxes_to_corners_3d", corners)
    cav_content = {"anchor_box": torch.ones((1, 1, 2, 7), dtype=torch.float32)}
    output = {
        "preds_dict_stage1": {
            "cls_preds": torch.tensor([[[[10.0]], [[-10.0]]]], dtype=torch.float32),
            "box_preds": torch.zeros((1, 14, 1, 1), dtype=torch.float32),
            "iou_preds": torch.tensor([[[[1.0]], [[-1.0]]]], dtype=torch.float32),
            "dir_cls_preds": torch.tensor(
                [[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]],
                dtype=torch.float32,
            ),
        }
    }

    boxes, scores = postprocessor.decode_agent_predictions(cav_content, output)

    assert boxes is expected_corners
    assert scores.shape == (1,)
    assert scores.item() == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item())
    selected_boxes = corners.call_args.args[0]
    assert selected_boxes.shape == (1, 7)
    assert selected_boxes[0, 6].item() == pytest.approx(0.25 + np.pi)
