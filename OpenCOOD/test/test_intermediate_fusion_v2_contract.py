"""Tests for the model-neutral IntermediateFusionDatasetV2 boundary."""

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

from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import (  # noqa: E402
    IntermediateFusionDatasetV2,
)
from opencood.models.communication_adapters import PoseFrameMetadata  # noqa: E402


class RecordingPayloadHandler:
    def __init__(self) -> None:
        self.published = []

    def set_opencda_payload(self, cav_id, module_name, payload) -> None:
        self.published.append((cav_id, module_name, payload))


def _build_dataset_for_extraction():
    dataset = IntermediateFusionDatasetV2.__new__(IntermediateFusionDatasetV2)
    dataset.cur_ego_pose_flag = True
    dataset.module_name = "OpenCOOD.IntermediateFusionDatasetV2"
    dataset.payload_handler = RecordingPayloadHandler()
    dataset.retrieve_base_data = MagicMock(
        return_value=OrderedDict(
            {
                "cav-1": {
                    "params": {"lidar_pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]},
                    "time_delay": 0,
                },
                "cav-2": {
                    "params": {"lidar_pose": [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]},
                    "time_delay": 2,
                },
            }
        )
    )
    dataset.build_agent_model_input = MagicMock(
        side_effect=lambda cav, pose: {
            "agent_pose": tuple(pose),
            "source": cav,
        }
    )
    return dataset


def test_extract_data_delegates_payload_construction_to_model_adapter():
    dataset = _build_dataset_for_extraction()
    builder = MagicMock(side_effect=lambda sample, metadata: (sample, metadata))

    dataset.extract_data(idx=10, agent_payload_builder=builder)

    assert builder.call_count == 2
    assert [entry[0] for entry in dataset.payload_handler.published] == ["cav-1", "cav-2"]
    assert all(entry[1] == "OpenCOOD.IntermediateFusionDatasetV2" for entry in dataset.payload_handler.published)
    first_metadata = dataset.payload_handler.published[0][2][1]
    second_metadata = dataset.payload_handler.published[1][2][1]
    assert isinstance(first_metadata, PoseFrameMetadata)
    assert first_metadata.capture_frame == 10
    assert second_metadata.capture_frame == 8
    assert first_metadata.lidar_pose == (1.0, 2.0, 0.0, 0.0, 0.0, 0.0)


def test_extract_data_does_not_assume_fpvrcnn_without_adapter():
    dataset = _build_dataset_for_extraction()

    with pytest.raises(NotImplementedError, match="model-specific communication adapter"):
        dataset.extract_data(idx=10)


def test_local_supervision_uses_complete_scene_and_stays_outside_wire_path():
    dataset = IntermediateFusionDatasetV2.__new__(IntermediateFusionDatasetV2)
    dataset.payload_handler = object()
    dataset.post_processor = MagicMock()
    object_centers = np.ones((4, 7), dtype=np.float32)
    object_mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    object_ids = ["vehicle-1", "vehicle-2"]
    anchors = np.ones((2, 2, 7), dtype=np.float32)
    dataset.post_processor.generate_object_center.return_value = (
        object_centers,
        object_mask,
        object_ids,
    )
    dataset.post_processor.generate_anchor_box.return_value = anchors
    base_data = OrderedDict(
        {
            "cav-1": {"objects": ["vehicle-1"]},
            "cav-2": {"objects": ["vehicle-2"]},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    supervision = dataset.build_local_supervision(base_data, ego_pose)

    dataset.post_processor.generate_object_center.assert_called_once_with(
        list(base_data.values()),
        ego_pose,
    )
    assert supervision["object_ids"] == object_ids
    np.testing.assert_array_equal(supervision["object_bbx_center"], object_centers)
    np.testing.assert_array_equal(supervision["object_bbx_mask"], object_mask)
    np.testing.assert_array_equal(supervision["anchor_box"], anchors)
    assert "stage2_label" not in supervision
    dataset.post_processor.generate_label.assert_not_called()


def test_local_supervision_builds_stage_two_labels_only_without_capi_transport():
    dataset = IntermediateFusionDatasetV2.__new__(IntermediateFusionDatasetV2)
    dataset.payload_handler = None
    dataset.post_processor = MagicMock()
    object_centers = np.ones((2, 7), dtype=np.float32)
    object_mask = np.ones(2, dtype=np.float32)
    anchors = np.ones((2, 2, 7), dtype=np.float32)
    stage_two_label = {"targets": np.ones((2, 2, 7), dtype=np.float32)}
    dataset.post_processor.generate_object_center.return_value = (
        object_centers,
        object_mask,
        ["vehicle-1", "vehicle-2"],
    )
    dataset.post_processor.generate_anchor_box.return_value = anchors
    dataset.post_processor.generate_label.return_value = stage_two_label

    supervision = dataset.build_local_supervision(
        OrderedDict({"cav-1": {}, "cav-2": {}}),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert supervision["stage2_label"] is stage_two_label
    dataset.post_processor.generate_label.assert_called_once_with(
        gt_box_center=object_centers,
        anchors=anchors,
        mask=object_mask,
    )


def test_received_payloads_are_decoded_by_attached_model_adapter():
    dataset = IntermediateFusionDatasetV2.__new__(IntermediateFusionDatasetV2)
    dataset.communication_adapter = MagicMock()
    dataset.communication_adapter.decode_received_payload.return_value = {"model_specific_feature": np.array([1.0], dtype=np.float32)}
    dataset.payload_handler = MagicMock()
    dataset.payload_handler.current_artery_payload = {"cav-1": {"cav-2": object()}}
    raw_payload = object()
    dataset.payload_handler.get_artery_payload.side_effect = [raw_payload, None]
    local_input = {
        "processed_features": {"voxel_features": np.ones((1, 4), dtype=np.float32)},
        "projected_lidar": np.ones((2, 4), dtype=np.float32),
    }
    dataset.build_agent_model_input = MagicMock(return_value=local_input)
    dataset.build_visualization_context = MagicMock(
        return_value={
            "projected_lidar_stack": [],
            "projected_lidar_roles": [],
            "projected_lidar_agent_ids": [],
        }
    )
    base_data = OrderedDict(
        {
            "cav-1": {"ego": True},
            "cav-2": {"ego": False},
            "cav-3": {"ego": False},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = dataset._IntermediateFusionDatasetV2__process_with_messages(
        "cav-1",
        ego_pose,
        base_data,
        ego_pose,
        base_data,
    )

    dataset.communication_adapter.decode_received_payload.assert_called_once_with(
        raw_payload,
        ego_pose,
    )
    assert result["processed_features"][0] is local_input["processed_features"]
    assert len(result["remote_agent_outputs"]) == 1
    np.testing.assert_array_equal(
        result["remote_agent_outputs"][0]["model_specific_feature"],
        np.array([1.0], dtype=np.float32),
    )
