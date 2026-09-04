"""Tests for model-owned communication adapters."""

from collections import OrderedDict
import sys
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

mocked_opencood = sys.modules.get("opencood")
if mocked_opencood is not None and not hasattr(mocked_opencood, "__path__"):
    pytest.skip(
        "full OpenCOOD tests are disabled when the lightweight OpenCDA test doubles are active",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from opencood.data_utils.datasets import early_fusion_dataset as early_dataset_module  # noqa: E402
from opencood.data_utils.datasets import late_fusion_dataset as late_dataset_module  # noqa: E402
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset  # noqa: E402
from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset  # noqa: E402
from opencood.models.communication_adapters import (  # noqa: E402
    BevInferenceInput,
    DetectionInferenceInput,
    EarlyFusionCommunicationAdapter,
    EarlyFusionWirePayload,
    FpvrcnnAgentInferenceInput,
    FpvrcnnWirePayload,
    IntermediateFeatureWirePayload,
    LateFusionWirePayload,
    ModelCommunicationAdapter,
    MultiScaleFeatureCommunicationAdapter,
    MultiScaleFeatureInput,
    PoseFrameMetadata,
    SpatialFeatureCommunicationAdapter,
    SpatialFeatureInput,
    Where2CommCommunicationAdapter,
    Where2CommWirePayload,
    build_communication_adapter,
)
from opencood.models.communication_adapters import early as early_module  # noqa: E402
from opencood.models.communication_adapters import fpvrcnn as fpvrcnn_module  # noqa: E402
from opencood.models.communication_adapters.fpvrcnn import FpvrcnnCommunicationAdapter  # noqa: E402
from opencood.models.communication_adapters.late import LateFusionCommunicationAdapter  # noqa: E402
from opencood.models.sub_modules.auto_encoder import AutoEncoder  # noqa: E402
from opencood.models.sub_modules.naive_compress import NaiveCompressor  # noqa: E402
from opencood.tools.inference_utils import inference_late_fusion  # noqa: E402


class DeclaredAdapter(ModelCommunicationAdapter):
    """Minimal declared adapter used to verify adapter selection."""

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        dataset.extract_data(idx)


class DeclaredModel(nn.Module):
    communication_adapter_class = DeclaredAdapter


class PlainModel(nn.Module):
    pass


class InvalidAdapterModel(nn.Module):
    communication_adapter_class = object


class FakePreprocessor:
    def __init__(self) -> None:
        self.received_features = None

    def collate_batch(self, features):
        self.received_features = features
        return {
            "voxel_features": torch.ones((2, 4), dtype=torch.float32),
            "voxel_coords": torch.zeros((2, 4), dtype=torch.int32),
            "voxel_num_points": torch.ones(2, dtype=torch.int32),
        }


class FakePostprocessor:
    @staticmethod
    def generate_anchor_box():
        return np.ones((2, 7), dtype=np.float32)


class FakeLatePostprocessor(FakePostprocessor):
    @staticmethod
    def decode_agent_predictions(model_input, output):
        assert "processed_lidar" in model_input
        assert output == {"head": "output"}
        return (
            torch.tensor(
                [[[1.0, 0.0, 2.0]] * 8],
                dtype=torch.float32,
            ),
            torch.tensor([0.75], dtype=torch.float32),
        )


class FakeFpvrcnnDataset:
    def __init__(self) -> None:
        self.pre_processor = FakePreprocessor()
        self.post_processor = FakePostprocessor()
        self.sample = {
            "processed_features": {
                "voxel_features": np.ones((2, 4), dtype=np.float32),
                "voxel_coords": np.zeros((2, 4), dtype=np.int32),
                "voxel_num_points": np.ones(2, dtype=np.int32),
            },
            "projected_lidar": np.array(
                [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.8]],
                dtype=np.float32,
            ),
        }
        self.metadata = PoseFrameMetadata(
            lidar_pose=(10.0, 20.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=12,
        )
        self.published_payload = None

    def extract_data(self, idx, agent_payload_builder=None) -> None:
        assert idx == 12
        assert agent_payload_builder is not None
        self.published_payload = agent_payload_builder(self.sample, self.metadata)


class FakeFpvrcnnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_batch = None

    def encode_agent(self, batch):
        self.received_batch = batch
        return {
            "proposals": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]],
                dtype=torch.float32,
            ),
            "proposal_scores": torch.tensor([0.75], dtype=torch.float32),
            "point_coords": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            "point_features": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        }


class FakeEarlyFusionDataset:
    def __init__(self) -> None:
        self.lidar_points = np.array(
            [[1.0, 0.0, 2.0, 0.5]],
            dtype=np.float32,
        )
        self.metadata = PoseFrameMetadata(
            lidar_pose=(10.0, 20.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=8,
        )
        self.published_payload = None

    def extract_data(self, idx, agent_payload_builder=None) -> None:
        assert idx == 8
        assert agent_payload_builder is not None
        self.published_payload = agent_payload_builder(self.lidar_points, self.metadata)


class FakeLateFusionDataset:
    def __init__(self) -> None:
        self.pre_processor = FakePreprocessor()
        self.post_processor = FakeLatePostprocessor()
        self.inference_input = BevInferenceInput(bev_input=np.ones((2, 3, 4), dtype=np.float32))
        self.metadata = PoseFrameMetadata(
            lidar_pose=(10.0, 20.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=8,
        )
        self.published_payload = None

    def extract_data(self, idx, agent_payload_builder) -> None:
        assert idx == 8
        self.published_payload = agent_payload_builder(
            self.inference_input,
            self.metadata,
        )


class FakeLateFusionModel(nn.Module):
    def forward(self, model_input):
        return {"head": "output"}


class FakeIntermediateDataset:
    def __init__(self) -> None:
        self.pre_processor = FakePreprocessor()
        self.inference_input = BevInferenceInput(bev_input=np.ones((2, 3, 4), dtype=np.float32))
        self.metadata = PoseFrameMetadata(
            lidar_pose=(10.0, 20.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=8,
        )
        self.published_payload = None

    def extract_data(self, idx, agent_payload_builder) -> None:
        assert idx == 8
        self.published_payload = agent_payload_builder(
            self.inference_input,
            self.metadata,
        )


class FakeSpatialFeatureModel(nn.Module):
    def encode_agent(self, model_input):
        assert "processed_lidar" in model_input
        return {
            "spatial_features": torch.arange(8, dtype=torch.float32).reshape(
                1,
                2,
                2,
                2,
            )
        }


class FakeMultiScaleFeatureModel(nn.Module):
    def encode_agent(self, model_input):
        assert "processed_lidar" in model_input
        return {
            "feature_maps": (
                torch.ones((1, 2, 2, 2), dtype=torch.float32),
                torch.full((1, 4, 1, 1), 2.0, dtype=torch.float32),
            )
        }


class FakeWhere2CommModel(nn.Module):
    def encode_agent(self, model_input, *, apply_communication_mask):
        assert "processed_lidar" in model_input
        value = 0.0 if apply_communication_mask else 1.0
        return {
            "spatial_features": torch.full(
                (1, 2, 2, 2),
                value,
                dtype=torch.float32,
            ),
            "communication_rate": torch.tensor(0.25),
        }


def test_build_communication_adapter_uses_model_declaration():
    model = DeclaredModel()

    adapter = build_communication_adapter(model, torch.device("cpu"))

    assert isinstance(adapter, DeclaredAdapter)
    assert adapter.model is model


def test_build_communication_adapter_requires_model_declaration():
    with pytest.raises(TypeError, match="must declare communication_adapter_class"):
        build_communication_adapter(PlainModel(), torch.device("cpu"))


def test_build_communication_adapter_selects_early_fusion_contract():
    model = PlainModel()

    adapter = build_communication_adapter(
        model,
        torch.device("cpu"),
        "EarlyFusionDataset",
    )

    assert isinstance(adapter, EarlyFusionCommunicationAdapter)
    assert adapter.model is model


def test_build_communication_adapter_selects_late_fusion_contract():
    model = PlainModel()

    adapter = build_communication_adapter(
        model,
        torch.device("cpu"),
        "LateFusionDataset",
    )

    assert isinstance(adapter, LateFusionCommunicationAdapter)
    assert adapter.model is model


def test_build_communication_adapter_rejects_invalid_declaration():
    with pytest.raises(TypeError, match="must derive from ModelCommunicationAdapter"):
        build_communication_adapter(InvalidAdapterModel(), torch.device("cpu"))


def test_early_adapter_sends_local_lidar_with_pose_metadata():
    dataset = FakeEarlyFusionDataset()
    adapter = EarlyFusionCommunicationAdapter(PlainModel(), torch.device("cpu"))

    adapter.prepare_transmission_payloads(dataset, idx=8)

    payload = dataset.published_payload
    assert isinstance(payload, EarlyFusionWirePayload)
    assert payload.metadata is dataset.metadata
    np.testing.assert_array_equal(payload.lidar_points, dataset.lidar_points)


def test_early_dataset_gives_adapter_sender_local_points(monkeypatch):
    monkeypatch.setattr(
        early_dataset_module.np.random,
        "permutation",
        np.arange,
    )
    dataset = object.__new__(EarlyFusionDataset)
    dataset.module_name = "OpenCOOD.EarlyFusionDataset"
    dataset.payload_handler = MagicMock()
    dataset.retrieve_base_data = Mock(
        return_value={
            "cav-1": {
                "lidar_np": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [4.0, 5.0, 6.0, 0.5],
                    ],
                    dtype=np.float32,
                ),
                "params": {"lidar_pose": [10.0, 20.0, 0.0, 0.0, 0.0, 0.0]},
                "time_delay": 2,
            }
        }
    )
    payload = object()
    payload_builder = Mock(return_value=payload)

    dataset.extract_data(10, payload_builder)

    lidar_points, metadata = payload_builder.call_args.args
    np.testing.assert_array_equal(
        lidar_points,
        np.array([[4.0, 5.0, 6.0, 0.5]], dtype=np.float32),
    )
    assert metadata == PoseFrameMetadata(
        lidar_pose=[10.0, 20.0, 0.0, 0.0, 0.0, 0.0],
        capture_frame=8,
    )
    dataset.payload_handler.set_opencda_payload.assert_called_once_with(
        "cav-1",
        dataset.module_name,
        payload,
    )


def test_early_adapter_projects_received_lidar_without_mutating_payload(monkeypatch):
    transformation = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(early_module, "x1_to_x2", lambda _sender, _receiver: transformation)
    local_points = np.array([[1.0, 0.0, 2.0, 0.5]], dtype=np.float32)
    payload = EarlyFusionWirePayload(
        lidar_points=local_points,
        metadata=PoseFrameMetadata(
            lidar_pose=(1.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=5,
        ),
    )
    adapter = EarlyFusionCommunicationAdapter(PlainModel(), torch.device("cpu"))

    decoded = adapter.decode_received_payload(
        payload,
        receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(decoded["projected_lidar"][0], [10.0, 21.0, 2.0, 0.5])
    np.testing.assert_array_equal(payload.lidar_points, local_points)


def test_early_adapter_rejects_payload_from_another_contract():
    adapter = EarlyFusionCommunicationAdapter(PlainModel(), torch.device("cpu"))

    with pytest.raises(TypeError, match="Expected EarlyFusionWirePayload"):
        adapter.decode_received_payload(
            object(),
            receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


def test_early_dataset_ignores_looped_back_ego_payload():
    dataset = object.__new__(EarlyFusionDataset)
    dataset.module_name = "OpenCOOD.EarlyFusionDataset"
    dataset.payload_handler = MagicMock()
    dataset.payload_handler.current_artery_payload = {"ego": object()}
    remote_payload = object()
    dataset.payload_handler.get_artery_payload.return_value = remote_payload
    dataset.communication_adapter = MagicMock()
    remote_lidar = np.full((1, 4), 2.0, dtype=np.float32)
    dataset.communication_adapter.decode_received_payload.return_value = {
        "projected_lidar": remote_lidar,
    }
    ego_lidar = np.ones((1, 4), dtype=np.float32)
    dataset.get_item_single_car = MagicMock(
        return_value={"projected_lidar": ego_lidar},
    )
    base_data = OrderedDict(
        {
            "ego": {"ego": True},
            "remote": {"ego": False},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = dataset._EarlyFusionDataset__process_with_messages(
        "ego",
        ego_pose,
        base_data,
    )

    assert len(result["projected_lidar_stack"]) == 2
    np.testing.assert_array_equal(result["projected_lidar_stack"][0], ego_lidar)
    np.testing.assert_array_equal(result["projected_lidar_stack"][1], remote_lidar)
    dataset.payload_handler.get_artery_payload.assert_called_once_with(
        "ego",
        "remote",
        dataset.module_name,
    )


def test_early_supervision_uses_complete_local_scene():
    dataset = object.__new__(EarlyFusionDataset)
    dataset.post_processor = MagicMock()
    boxes = np.ones((2, 7), dtype=np.float32)
    mask = np.ones(2, dtype=np.float32)
    object_ids = ["vehicle-1", "vehicle-2"]
    dataset.post_processor.generate_object_center.return_value = (
        boxes,
        mask,
        object_ids,
    )
    base_data = OrderedDict(
        {
            "ego": {"objects": ["vehicle-1"]},
            "remote": {"objects": ["vehicle-2"]},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    supervision = dataset.build_local_supervision(base_data, ego_pose)

    dataset.post_processor.generate_object_center.assert_called_once_with(
        list(base_data.values()),
        ego_pose,
    )
    assert supervision["object_bbx_center"] is boxes
    assert supervision["object_bbx_mask"] is mask
    assert supervision["object_ids"] is object_ids


def test_late_adapter_sends_decoded_detection_candidates():
    dataset = FakeLateFusionDataset()
    adapter = LateFusionCommunicationAdapter(
        FakeLateFusionModel(),
        torch.device("cpu"),
    )

    adapter.prepare_transmission_payloads(dataset, idx=8)

    payload = dataset.published_payload
    assert isinstance(payload, LateFusionWirePayload)
    assert payload.metadata is dataset.metadata
    assert payload.detections.boxes.shape == (1, 8, 3)
    np.testing.assert_array_equal(
        payload.detections.scores,
        np.array([0.75], dtype=np.float32),
    )


def test_late_adapter_projects_received_3d_detections(monkeypatch):
    transformation = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        "opencood.models.communication_adapters.late.x1_to_x2",
        lambda _sender, _receiver: transformation,
    )
    local_boxes = np.array([[[1.0, 0.0, 2.0]] * 8], dtype=np.float32)
    payload = LateFusionWirePayload(
        detections=DetectionInferenceInput(
            boxes=local_boxes,
            scores=np.array([0.9], dtype=np.float32),
        ),
        metadata=PoseFrameMetadata(
            lidar_pose=(1.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=5,
        ),
    )
    adapter = LateFusionCommunicationAdapter(PlainModel(), torch.device("cpu"))

    decoded = adapter.decode_received_payload(
        payload,
        receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(decoded["pred_box_tensor"][0, 0], [10.0, 21.0, 2.0])
    np.testing.assert_allclose(decoded["pred_score"], [0.9])
    np.testing.assert_array_equal(payload.detections.boxes, local_boxes)


def test_late_adapter_projects_received_2d_detections(monkeypatch):
    transformation = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        "opencood.models.communication_adapters.late.x1_to_x2",
        lambda _sender, _receiver: transformation,
    )
    payload = LateFusionWirePayload(
        detections=DetectionInferenceInput(
            boxes=np.array(
                [[[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]],
                dtype=np.float32,
            ),
            scores=np.array([0.9], dtype=np.float32),
        ),
        metadata=PoseFrameMetadata(
            lidar_pose=(1.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=5,
        ),
    )
    adapter = LateFusionCommunicationAdapter(PlainModel(), torch.device("cpu"))

    decoded = adapter.decode_received_payload(
        payload,
        receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        decoded["pred_box_tensor"],
        np.array(
            [[[11.0, 22.0], [13.0, 22.0], [13.0, 24.0], [11.0, 24.0]]],
            dtype=np.float32,
        ),
    )


def test_late_inference_does_not_run_model_for_received_detections():
    local_input = {"processed_lidar": object()}
    remote_detections = {
        "pred_box_tensor": torch.ones((1, 8, 3)),
        "pred_score": torch.tensor([0.8]),
    }
    batch = {"ego": local_input, "cav-2": remote_detections}
    local_output = {"psm": object(), "rm": object()}
    model = Mock(return_value=local_output)
    expected = (object(), object(), object())
    dataset = MagicMock()
    dataset.post_process.return_value = expected

    result = inference_late_fusion(batch, model, dataset)

    assert result == expected
    model.assert_called_once_with(local_input)
    dataset.post_process.assert_called_once()
    post_process_batch, output_dict = dataset.post_process.call_args.args
    assert post_process_batch is batch
    assert output_dict == {"ego": local_output}


def test_late_dataset_ignores_looped_back_ego_payload():
    dataset = object.__new__(LateFusionDataset)
    dataset.module_name = "OpenCOOD.LateFusionDataset"
    dataset.payload_handler = MagicMock()
    dataset.payload_handler.current_artery_payload = {"ego": object()}
    remote_payload = object()
    dataset.payload_handler.get_artery_payload.return_value = remote_payload
    remote_detections = {
        "pred_box_tensor": np.ones((1, 8, 3), dtype=np.float32),
        "pred_score": np.ones(1, dtype=np.float32),
    }
    dataset.communication_adapter = MagicMock()
    dataset.communication_adapter.decode_received_payload.return_value = remote_detections
    local_input = object()
    dataset._LateFusionDataset__build_model_input = MagicMock(
        return_value={"inference_input": local_input},
    )
    base_data = OrderedDict(
        {
            "ego": {"ego": True},
            "remote": {"ego": False},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = dataset._LateFusionDataset__process_with_messages(
        "ego",
        ego_pose,
        base_data,
    )

    assert result["ego"]["inference_input"] is local_input
    assert result["remote"] is remote_detections
    dataset.payload_handler.get_artery_payload.assert_called_once_with(
        "ego",
        "remote",
        dataset.module_name,
    )


def test_late_dataset_merges_local_and_delivered_detections_before_global_nms(
    monkeypatch,
):
    monkeypatch.setattr(
        late_dataset_module.box_utils,
        "project_box3d",
        lambda boxes, _transformation: boxes,
    )
    dataset = object.__new__(LateFusionDataset)
    local_boxes = torch.ones((1, 8, 3), dtype=torch.float32)
    remote_boxes = torch.full((1, 8, 3), 2.0, dtype=torch.float32)
    local_scores = torch.tensor([0.9], dtype=torch.float32)
    remote_scores = torch.tensor([0.8], dtype=torch.float32)
    expected_boxes = torch.full((1, 8, 3), 3.0, dtype=torch.float32)
    expected_scores = torch.tensor([0.7], dtype=torch.float32)
    expected_gt = object()
    dataset.post_processor = MagicMock()
    dataset.post_processor.decode_agent_predictions.return_value = (
        local_boxes,
        local_scores,
    )
    dataset.post_processor.post_process_detections.return_value = (
        expected_boxes,
        expected_scores,
    )
    dataset.post_processor.generate_gt_bbx.return_value = expected_gt
    batch = OrderedDict(
        {
            "ego": {
                "processed_lidar": object(),
                "transformation_matrix": torch.eye(4),
            },
            "remote": {
                "pred_box_tensor": remote_boxes,
                "pred_score": remote_scores,
            },
        }
    )

    result = dataset.post_process(batch, {"ego": {"model": "output"}})

    assert result[0] is expected_boxes
    assert result[1] is expected_scores
    assert result[2] is expected_gt
    merged_boxes, merged_scores = dataset.post_processor.post_process_detections.call_args.args
    torch.testing.assert_close(
        merged_boxes,
        torch.cat([local_boxes, remote_boxes], dim=0),
    )
    torch.testing.assert_close(
        merged_scores,
        torch.cat([local_scores, remote_scores], dim=0),
    )
    dataset.post_processor.decode_agent_predictions.assert_called_once_with(
        batch["ego"],
        {"model": "output"},
    )


def test_late_supervision_and_labels_use_complete_local_scene():
    dataset = object.__new__(LateFusionDataset)
    dataset.post_processor = MagicMock()
    boxes = np.ones((2, 7), dtype=np.float32)
    mask = np.ones(2, dtype=np.float32)
    object_ids = ["vehicle-1", "vehicle-2"]
    anchors = np.full((2, 7), 2.0, dtype=np.float32)
    labels = {"targets": np.full((2, 7), 3.0, dtype=np.float32)}
    dataset.post_processor.generate_object_center.return_value = (
        boxes,
        mask,
        object_ids,
    )
    dataset.post_processor.generate_anchor_box.return_value = anchors
    dataset.post_processor.generate_label.return_value = labels
    base_data = OrderedDict(
        {
            "ego": {"objects": ["vehicle-1"]},
            "remote": {"objects": ["vehicle-2"]},
        }
    )
    ego_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    supervision = dataset.build_local_supervision(base_data, ego_pose)

    dataset.post_processor.generate_object_center.assert_called_once_with(
        list(base_data.values()),
        ego_pose,
    )
    dataset.post_processor.generate_label.assert_called_once_with(
        gt_box_center=boxes,
        anchors=anchors,
        mask=mask,
    )
    assert supervision["object_ids"] == object_ids
    assert supervision["label_dict"] is labels
    np.testing.assert_array_equal(supervision["anchor_box"], anchors)


def test_spatial_feature_adapter_sends_learned_feature_map():
    dataset = FakeIntermediateDataset()
    adapter = SpatialFeatureCommunicationAdapter(
        FakeSpatialFeatureModel(),
        torch.device("cpu"),
    )

    adapter.prepare_transmission_payloads(dataset, idx=8)

    payload = dataset.published_payload
    assert isinstance(payload, IntermediateFeatureWirePayload)
    assert isinstance(payload.inference_input, SpatialFeatureInput)
    assert payload.metadata is dataset.metadata
    np.testing.assert_array_equal(
        payload.inference_input.spatial_features,
        np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
    )


def test_multiscale_feature_adapter_preserves_ordered_scales():
    dataset = FakeIntermediateDataset()
    adapter = MultiScaleFeatureCommunicationAdapter(
        FakeMultiScaleFeatureModel(),
        torch.device("cpu"),
    )

    adapter.prepare_transmission_payloads(dataset, idx=8)
    payload = dataset.published_payload
    decoded = adapter.decode_received_payload(
        payload,
        receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert isinstance(payload, IntermediateFeatureWirePayload)
    assert isinstance(payload.inference_input, MultiScaleFeatureInput)
    np.testing.assert_array_equal(decoded["feature_0"], np.ones((1, 2, 2, 2)))
    np.testing.assert_array_equal(decoded["feature_1"], np.full((1, 4, 1, 1), 2.0))
    assert decoded["metadata"] is dataset.metadata


def test_where2comm_adapter_masks_only_transmitted_features():
    dataset = FakeIntermediateDataset()
    adapter = Where2CommCommunicationAdapter(
        FakeWhere2CommModel(),
        torch.device("cpu"),
    )

    adapter.prepare_transmission_payloads(dataset, idx=8)
    payload = dataset.published_payload
    local_features = adapter.encode_local_intermediate_input(
        dataset,
        dataset.inference_input,
    )

    assert isinstance(payload, Where2CommWirePayload)
    np.testing.assert_array_equal(
        payload.inference_input.spatial_features,
        np.zeros((1, 2, 2, 2)),
    )
    np.testing.assert_array_equal(
        local_features["spatial_features"],
        np.ones((1, 2, 2, 2)),
    )
    assert payload.inference_input.communication_rate == pytest.approx(0.25)


def test_channel_compressor_exposes_smaller_wire_representation():
    compressor = NaiveCompressor(input_dim=8, compress_raito=2).eval()
    spatial_features = torch.ones((1, 8, 4, 4), dtype=torch.float32)

    encoded = compressor.encode(spatial_features)
    decoded = compressor.decode(encoded)

    assert encoded.shape == (1, 4, 4, 4)
    assert decoded.shape == spatial_features.shape
    torch.testing.assert_close(compressor(spatial_features), decoded)


def test_autoencoder_exposes_smaller_wire_representation():
    compressor = AutoEncoder(feature_num=8, layer_num=1).eval()
    spatial_features = torch.ones((1, 8, 4, 4), dtype=torch.float32)

    encoded = compressor.encode(spatial_features)
    decoded = compressor.decode(encoded)

    assert encoded.shape == (1, 4, 2, 2)
    assert decoded.shape == spatial_features.shape
    torch.testing.assert_close(compressor(spatial_features), decoded)


def test_fpvrcnn_adapter_builds_payload_from_sender_model_outputs():
    model = FakeFpvrcnnModel()
    dataset = FakeFpvrcnnDataset()
    adapter = FpvrcnnCommunicationAdapter(model, torch.device("cpu"))

    adapter.prepare_transmission_payloads(dataset, idx=12)

    payload = dataset.published_payload
    assert isinstance(payload, FpvrcnnWirePayload)
    assert payload.metadata is dataset.metadata
    np.testing.assert_array_equal(
        payload.inference_input.proposals,
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        payload.inference_input.point_features,
        np.array([[0.1, 0.2]], dtype=np.float32),
    )
    assert dataset.pre_processor.received_features[0] is dataset.sample["processed_features"]
    assert model.training is False
    np.testing.assert_array_equal(
        model.received_batch["origin_lidar"].cpu().numpy(),
        np.array(
            [[0.0, 1.0, 2.0, 3.0, 0.5], [0.0, 4.0, 5.0, 6.0, 0.8]],
            dtype=np.float32,
        ),
    )


def test_fpvrcnn_adapter_projects_received_outputs_without_mutating_payload(monkeypatch):
    transformation = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(fpvrcnn_module, "x1_to_x2", lambda _sender, _receiver: transformation)
    original_proposals = np.array(
        [[1.0, 0.0, 2.0, 4.0, 5.0, 6.0, 0.25]],
        dtype=np.float32,
    )
    original_points = np.array([[1.0, 0.0, 2.0]], dtype=np.float32)
    payload = FpvrcnnWirePayload(
        inference_input=FpvrcnnAgentInferenceInput(
            proposals=original_proposals,
            proposal_scores=np.array([0.9], dtype=np.float32),
            point_coords=original_points,
            point_features=np.array([[0.1, 0.2]], dtype=np.float32),
        ),
        metadata=PoseFrameMetadata(
            lidar_pose=(1.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            capture_frame=5,
        ),
    )
    adapter = FpvrcnnCommunicationAdapter(FakeFpvrcnnModel(), torch.device("cpu"))

    decoded = adapter.decode_received_payload(
        payload,
        receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(decoded["proposals"][0, :3], [10.0, 21.0, 2.0])
    np.testing.assert_allclose(decoded["proposals"][0, 6], 0.25 + np.pi / 2)
    np.testing.assert_allclose(decoded["point_coords"][0], [10.0, 21.0, 2.0])
    np.testing.assert_array_equal(payload.inference_input.proposals, original_proposals)
    np.testing.assert_array_equal(payload.inference_input.point_coords, original_points)


def test_fpvrcnn_adapter_rejects_payload_from_another_contract():
    adapter = FpvrcnnCommunicationAdapter(FakeFpvrcnnModel(), torch.device("cpu"))

    with pytest.raises(TypeError, match="Expected FpvrcnnWirePayload"):
        adapter.decode_received_payload(
            object(),
            receiver_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
