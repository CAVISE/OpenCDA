"""Tests for typed cooperative-perception communication contracts."""

from dataclasses import fields
import pickle

import numpy as np
import pytest

from opencood.models.communication_adapters import (
    BevInferenceInput,
    EarlyFusionWirePayload,
    FpvrcnnAgentInferenceInput,
    FpvrcnnWirePayload,
    IntermediateFusionWirePayload,
    LateFusionWirePayload,
    PointCloudInferenceInput,
    PoseFrameMetadata,
    VoxelInferenceInput,
    build_inference_input,
    inference_input_to_dict,
    merge_inference_inputs,
)


@pytest.mark.parametrize(
    ("preprocessor_output", "expected_type"),
    [
        (
            {
                "voxel_features": np.ones((2, 4), dtype=np.float32),
                "voxel_coords": np.zeros((2, 4), dtype=np.int32),
            },
            VoxelInferenceInput,
        ),
        (
            {
                "voxel_features": np.ones((2, 4), dtype=np.float32),
                "voxel_coords": np.zeros((2, 4), dtype=np.int32),
                "voxel_num_points": np.ones(2, dtype=np.int32),
            },
            VoxelInferenceInput,
        ),
        ({"bev_input": np.ones((2, 3, 4), dtype=np.float32)}, BevInferenceInput),
        (
            {"downsample_lidar": np.ones((8, 4), dtype=np.float32)},
            PointCloudInferenceInput,
        ),
    ],
)
def test_build_inference_input_uses_actual_preprocessor_fields(
    preprocessor_output,
    expected_type,
):
    inference_input = build_inference_input(preprocessor_output)

    assert isinstance(inference_input, expected_type)
    converted = inference_input_to_dict(inference_input)
    assert converted.keys() == preprocessor_output.keys()
    for field_name, value in preprocessor_output.items():
        np.testing.assert_array_equal(converted[field_name], value)


def test_build_inference_input_rejects_unknown_or_mixed_fields():
    with pytest.raises(ValueError, match="Unsupported preprocessor output fields"):
        build_inference_input(
            {
                "bev_input": np.ones((2, 2), dtype=np.float32),
                "voxel_features": np.ones((2, 2), dtype=np.float32),
            }
        )


def test_merge_inference_inputs_preserves_all_agent_values():
    first = BevInferenceInput(bev_input=np.full((2, 2), 1.0, dtype=np.float32))
    second = BevInferenceInput(bev_input=np.full((2, 2), 2.0, dtype=np.float32))

    merged = merge_inference_inputs([first, second])

    assert list(merged) == ["bev_input"]
    np.testing.assert_array_equal(merged["bev_input"][0], first.bev_input)
    np.testing.assert_array_equal(merged["bev_input"][1], second.bev_input)


def test_merge_inference_inputs_rejects_heterogeneous_representations():
    with pytest.raises(TypeError, match="same inference input representation"):
        merge_inference_inputs(
            [
                BevInferenceInput(bev_input=np.ones((2, 2), dtype=np.float32)),
                PointCloudInferenceInput(points=np.ones((2, 4), dtype=np.float32)),
            ]
        )


def test_wire_payloads_expose_only_network_contract_fields():
    forbidden_fields = {
        "anchor_box",
        "label_dict",
        "neg_equal_one",
        "object_bbx_center",
        "object_bbx_mask",
        "object_ids",
        "pos_equal_one",
        "schema_version",
        "spoofing_mask",
        "targets",
    }
    expected_fields = {
        EarlyFusionWirePayload: {"projected_lidar"},
        LateFusionWirePayload: {"inference_input", "metadata"},
        IntermediateFusionWirePayload: {"inference_input", "metadata"},
        FpvrcnnWirePayload: {"inference_input", "metadata"},
    }

    for payload_type, expected in expected_fields.items():
        payload_fields = {field.name for field in fields(payload_type)}
        assert payload_fields == expected
        assert payload_fields.isdisjoint(forbidden_fields)


def test_fpvrcnn_wire_payload_contains_stage_one_and_vsa_outputs():
    metadata = PoseFrameMetadata(lidar_pose=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0), capture_frame=17)
    inference_input = FpvrcnnAgentInferenceInput(
        proposals=np.ones((2, 7), dtype=np.float32),
        proposal_scores=np.array([0.9, 0.8], dtype=np.float32),
        point_coords=np.ones((3, 3), dtype=np.float32),
        point_features=np.ones((3, 8), dtype=np.float32),
    )

    payload = FpvrcnnWirePayload(inference_input=inference_input, metadata=metadata)

    assert payload.metadata.capture_frame == 17
    assert payload.inference_input.proposals.shape == (2, 7)
    assert payload.inference_input.point_features.shape == (3, 8)


def test_typed_wire_payload_survives_v2x_pickle_round_trip():
    payload = LateFusionWirePayload(
        inference_input=BevInferenceInput(bev_input=np.arange(12, dtype=np.float32).reshape(3, 4)),
        metadata=PoseFrameMetadata(
            lidar_pose=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
            capture_frame=21,
        ),
    )

    restored = pickle.loads(pickle.dumps(payload))

    assert isinstance(restored, LateFusionWirePayload)
    assert restored.metadata == payload.metadata
    np.testing.assert_array_equal(
        restored.inference_input.bev_input,
        payload.inference_input.bev_input,
    )
