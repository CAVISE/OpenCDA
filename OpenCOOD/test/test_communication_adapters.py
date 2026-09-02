"""Tests for model-owned communication adapters."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from opencood.models.communication_adapters import (
    FpvrcnnAgentInferenceInput,
    FpvrcnnWirePayload,
    ModelCommunicationAdapter,
    PendingModelCommunicationAdapter,
    PoseFrameMetadata,
    build_communication_adapter,
)
from opencood.models.communication_adapters import fpvrcnn as fpvrcnn_module
from opencood.models.communication_adapters.fpvrcnn import FpvrcnnCommunicationAdapter


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


def test_build_communication_adapter_uses_model_declaration():
    model = DeclaredModel()

    adapter = build_communication_adapter(model, torch.device("cpu"))

    assert isinstance(adapter, DeclaredAdapter)
    assert adapter.model is model


def test_build_communication_adapter_uses_pending_fallback():
    adapter = build_communication_adapter(PlainModel(), torch.device("cpu"))

    assert isinstance(adapter, PendingModelCommunicationAdapter)


def test_build_communication_adapter_rejects_invalid_declaration():
    with pytest.raises(TypeError, match="must derive from ModelCommunicationAdapter"):
        build_communication_adapter(InvalidAdapterModel(), torch.device("cpu"))


def test_pending_adapter_delegates_to_dataset_payload_path():
    dataset = MagicMock()
    adapter = PendingModelCommunicationAdapter(PlainModel(), torch.device("cpu"))

    adapter.prepare_transmission_payloads(dataset, idx=4)

    dataset.extract_data.assert_called_once_with(4)


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
