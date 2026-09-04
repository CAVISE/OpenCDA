"""Tests for the learned-feature Intermediate fusion boundary."""

from collections import OrderedDict
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

pytest.importorskip("torch")
pytest.importorskip("open3d")

from opencood.data_utils.datasets.intermediate_fusion_dataset import (  # noqa: E402
    IntermediateFusionDataset,
)
from opencood.models.communication_adapters import (  # noqa: E402
    PoseFrameMetadata,
    V2XViTMetadata,
)


class RecordingPayloadHandler:
    """Collect payloads published by a dataset extraction call."""

    def __init__(self) -> None:
        self.published = []

    def set_opencda_payload(self, cav_id, module_name, payload) -> None:
        self.published.append((cav_id, module_name, payload))


def test_extract_data_requires_adapter_to_build_learned_payloads():
    dataset = IntermediateFusionDataset.__new__(IntermediateFusionDataset)
    dataset.cur_ego_pose_flag = True
    dataset.module_name = "OpenCOOD.IntermediateFusionDataset"
    dataset.model_name = "point_pillar_fcooper"
    dataset.payload_handler = RecordingPayloadHandler()
    dataset.retrieve_base_data = MagicMock(
        return_value=OrderedDict(
            {
                "cav-1": {
                    "ego": True,
                    "params": {"lidar_pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]},
                    "time_delay": 0,
                },
                "cav-2": {
                    "ego": False,
                    "params": {"lidar_pose": [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]},
                    "time_delay": 1,
                },
            }
        )
    )
    inference_inputs = [object(), object()]
    dataset.get_item_single_car = MagicMock(
        side_effect=[
            {"inference_input": inference_inputs[0]},
            {"inference_input": inference_inputs[1]},
        ]
    )
    payload_builder = MagicMock(side_effect=lambda model_input, metadata: (model_input, metadata))

    dataset.extract_data(7, payload_builder)

    assert payload_builder.call_count == 2
    assert [entry[0] for entry in dataset.payload_handler.published] == [
        "cav-1",
        "cav-2",
    ]
    assert dataset.payload_handler.published[0][2] == (inference_inputs[0], None)
    assert dataset.payload_handler.published[1][2] == (inference_inputs[1], None)


def test_message_assembly_keeps_only_delivered_learned_features():
    dataset = IntermediateFusionDataset.__new__(IntermediateFusionDataset)
    dataset.communication_adapter = MagicMock()
    dataset.communication_adapter.encode_local_intermediate_input.return_value = {"spatial_features": np.ones((1, 2, 2, 2), dtype=np.float32)}
    dataset.communication_adapter.decode_received_payload.return_value = {
        "spatial_features": np.full((1, 2, 2, 2), 2.0, dtype=np.float32),
        "metadata": None,
    }
    dataset.payload_handler = MagicMock()
    dataset.payload_handler.current_artery_payload = {"cav-1": object()}
    delivered_payload = object()
    dataset.payload_handler.get_artery_payload.side_effect = lambda _receiver, sender, _module: delivered_payload if sender == "cav-2" else None
    dataset.module_name = "OpenCOOD.IntermediateFusionDataset"
    dataset.get_item_single_car = MagicMock(return_value={"inference_input": object()})
    dataset.build_model_metadata = MagicMock(return_value=None)
    base_data = OrderedDict(
        {
            "cav-1": {"ego": True},
            "cav-2": {"ego": False},
            "cav-3": {"ego": False},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = dataset._IntermediateFusionDataset__process_with_messages(
        "cav-1",
        ego_pose,
        base_data,
        receive_frame=9,
    )

    assert len(result["intermediate_features"]) == 2
    assert result["metadata"] == [None, None]
    dataset.communication_adapter.decode_received_payload.assert_called_once_with(
        delivered_payload,
        ego_pose,
    )


def test_inference_sample_concatenates_agent_feature_batches():
    dataset = IntermediateFusionDataset.__new__(IntermediateFusionDataset)
    dataset.model_name = "point_pillar_fcooper"
    dataset.visualize = False
    local_supervision = {
        "object_bbx_center": np.ones((2, 7), dtype=np.float32),
        "object_bbx_mask": np.ones(2, dtype=np.float32),
        "object_ids": ["vehicle-1"],
        "anchor_box": np.ones((2, 7), dtype=np.float32),
        "label_dict": {},
    }

    sample = dataset.assemble_inference_sample(
        inference_input={
            "intermediate_features": [
                {"spatial_features": np.ones((1, 2, 2, 2), dtype=np.float32)},
                {"spatial_features": np.full((1, 2, 2, 2), 2.0, dtype=np.float32)},
            ],
            "metadata": [None, None],
        },
        local_supervision=local_supervision,
        visualization_context={},
        receive_frame=3,
    )

    ego_sample = sample["ego"]
    assert ego_sample["cav_num"] == 2
    assert "processed_lidar" not in ego_sample
    np.testing.assert_array_equal(
        ego_sample["intermediate_features"]["spatial_features"][:, 0, 0, 0],
        np.array([1.0, 2.0], dtype=np.float32),
    )


def test_pairwise_transforms_are_derived_only_for_available_agents(monkeypatch):
    dataset = IntermediateFusionDataset.__new__(IntermediateFusionDataset)
    dataset.model_name = "point_pillar_v2vnet"
    dataset.max_cav = 3
    dataset.proj_first = False
    dataset.visualize = False
    transform = MagicMock(
        side_effect=lambda source, target: np.full(
            (4, 4),
            source[0] * 10 + target[0],
            dtype=np.float64,
        )
    )
    monkeypatch.setattr(
        "opencood.data_utils.datasets.intermediate_fusion_dataset.x1_to_x2",
        transform,
    )
    metadata = [
        PoseFrameMetadata(
            lidar_pose=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=4,
        ),
        PoseFrameMetadata(
            lidar_pose=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=4,
        ),
    ]

    sample = dataset.assemble_inference_sample(
        inference_input={
            "intermediate_features": [
                {"spatial_features": np.ones((1, 2, 2, 2), dtype=np.float32)},
                {"spatial_features": np.ones((1, 2, 2, 2), dtype=np.float32)},
            ],
            "metadata": metadata,
        },
        local_supervision={},
        visualization_context={},
        receive_frame=4,
    )

    pairwise = sample["ego"]["pairwise_t_matrix"]
    assert pairwise.shape == (3, 3, 4, 4)
    assert transform.call_count == 4
    np.testing.assert_array_equal(pairwise[0, 1], np.full((4, 4), 12.0))
    np.testing.assert_array_equal(pairwise[2], np.zeros((3, 4, 4)))


def test_v2xvit_metadata_is_padded_after_delivered_agents():
    dataset = IntermediateFusionDataset.__new__(IntermediateFusionDataset)
    dataset.model_name = "point_pillar_transformer"
    dataset.max_cav = 3
    dataset.visualize = False
    first_correction = np.eye(4, dtype=np.float32)
    second_correction = np.full((4, 4), 2.0, dtype=np.float32)
    metadata = [
        V2XViTMetadata(
            velocity=0.5,
            agent_type=0,
            capture_frame=8,
            spatial_correction_matrix=first_correction,
        ),
        V2XViTMetadata(
            velocity=0.25,
            agent_type=1,
            capture_frame=6,
            spatial_correction_matrix=second_correction,
        ),
    ]

    sample = dataset.assemble_inference_sample(
        inference_input={
            "intermediate_features": [
                {"spatial_features": np.ones((1, 2, 2, 2), dtype=np.float32)},
                {"spatial_features": np.ones((1, 2, 2, 2), dtype=np.float32)},
            ],
            "metadata": metadata,
        },
        local_supervision={},
        visualization_context={},
        receive_frame=10,
    )

    ego_sample = sample["ego"]
    assert ego_sample["velocity"] == [0.5, 0.25, 0.0]
    assert ego_sample["time_delay"] == [2.0, 4.0, 0.0]
    assert ego_sample["infra"] == [0, 1, 0.0]
    np.testing.assert_array_equal(
        ego_sample["spatial_correction_matrix"],
        np.stack([first_correction, second_correction, np.eye(4)]),
    )
