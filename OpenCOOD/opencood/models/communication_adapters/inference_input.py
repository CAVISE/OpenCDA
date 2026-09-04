"""Typed model inputs exchanged through communication adapters."""

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

import numpy.typing as npt


VOXEL_INPUT_FIELDS = frozenset({"voxel_features", "voxel_coords"})
SPARSE_VOXEL_INPUT_FIELDS = VOXEL_INPUT_FIELDS | {"voxel_num_points"}
BEV_INPUT_FIELDS = frozenset({"bev_input"})
POINT_CLOUD_INPUT_FIELDS = frozenset({"downsample_lidar"})


@dataclass(frozen=True, slots=True, kw_only=True)
class VoxelInferenceInput:
    """
    Voxelized LiDAR input.

    Attributes
    ----------
    voxel_features : numpy.typing.NDArray[Any]
        Features stored for every non-empty voxel.
    voxel_coords : numpy.typing.NDArray[Any]
        Coordinates of the non-empty voxels.
    voxel_num_points : numpy.typing.NDArray[Any] | None
        Number of points in every voxel when supplied by the preprocessor.
    """

    voxel_features: npt.NDArray[Any]
    voxel_coords: npt.NDArray[Any]
    voxel_num_points: npt.NDArray[Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BevInferenceInput:
    """
    Bird's-eye-view model input.

    Attributes
    ----------
    bev_input : numpy.typing.NDArray[Any]
        Dense BEV feature map produced by the preprocessor.
    """

    bev_input: npt.NDArray[Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class PointCloudInferenceInput:
    """
    Point-cloud model input.

    Attributes
    ----------
    points : numpy.typing.NDArray[Any]
        Sampled LiDAR points produced by the preprocessor.
    """

    points: npt.NDArray[Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class FpvrcnnAgentInferenceInput:
    """
    Agent-side FPV-RCNN output consumed by cooperative stage two.

    Attributes
    ----------
    proposals : numpy.typing.NDArray[Any]
        Stage-one bounding-box proposals in the sender coordinate frame.
    proposal_scores : numpy.typing.NDArray[Any]
        Confidence scores corresponding to ``proposals``.
    point_coords : numpy.typing.NDArray[Any]
        Coordinates of VSA keypoints in the sender coordinate frame.
    point_features : numpy.typing.NDArray[Any]
        VSA features corresponding to ``point_coords``.
    """

    proposals: npt.NDArray[Any]
    proposal_scores: npt.NDArray[Any]
    point_coords: npt.NDArray[Any]
    point_features: npt.NDArray[Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectionInferenceInput:
    """
    Detection candidates produced by one late-fusion sender.

    Attributes
    ----------
    boxes : numpy.typing.NDArray[Any]
        Bounding-box corners expressed in the sender coordinate frame.
    scores : numpy.typing.NDArray[Any]
        Confidence scores corresponding to ``boxes``.
    """

    boxes: npt.NDArray[Any]
    scores: npt.NDArray[Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class Where2CommFeatureInput:
    """
    Learned feature data selected by a Where2Comm sender.

    Attributes
    ----------
    spatial_features : numpy.typing.NDArray[Any]
        Sender-side feature representation after the communication mask and
        optional learned compression encoder.
    communication_rate : float
        Fraction of spatial locations selected for transmission.
    """

    spatial_features: npt.NDArray[Any]
    communication_rate: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialFeatureInput:
    """
    Single learned spatial feature map produced by an agent encoder.

    Attributes
    ----------
    spatial_features : numpy.typing.NDArray[Any]
        Feature map including its single-agent batch dimension.
    """

    spatial_features: npt.NDArray[Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiScaleFeatureInput:
    """
    Learned feature maps produced at multiple encoder scales.

    Attributes
    ----------
    feature_maps : tuple[numpy.typing.NDArray[Any], ...]
        Ordered feature maps including their single-agent batch dimensions.
    """

    feature_maps: tuple[npt.NDArray[Any], ...]


InferenceInput: TypeAlias = VoxelInferenceInput | BevInferenceInput | PointCloudInferenceInput


def build_inference_input(preprocessor_output: Mapping[str, Any]) -> InferenceInput:
    """
    Convert a preprocessor result into a typed inference input.

    The concrete type is selected from the fields actually returned by the
    preprocessor. Dataset and transport code therefore remain independent of
    the configured preprocessor class.

    Parameters
    ----------
    preprocessor_output : collections.abc.Mapping[str, Any]
        Result returned by ``preprocessor.preprocess()``.

    Returns
    -------
    InferenceInput
        Typed representation of the preprocessor result.

    Raises
    ------
    ValueError
        If the returned field set does not match a supported input type.
    """
    fields = frozenset(preprocessor_output)

    if fields == VOXEL_INPUT_FIELDS or fields == SPARSE_VOXEL_INPUT_FIELDS:
        return VoxelInferenceInput(
            voxel_features=preprocessor_output["voxel_features"],
            voxel_coords=preprocessor_output["voxel_coords"],
            voxel_num_points=preprocessor_output.get("voxel_num_points"),
        )
    if fields == BEV_INPUT_FIELDS:
        return BevInferenceInput(bev_input=preprocessor_output["bev_input"])
    if fields == POINT_CLOUD_INPUT_FIELDS:
        return PointCloudInferenceInput(points=preprocessor_output["downsample_lidar"])

    raise ValueError(f"Unsupported preprocessor output fields: {sorted(fields)}")


def inference_input_to_dict(inference_input: InferenceInput) -> dict[str, npt.NDArray[Any]]:
    """
    Convert a typed inference input to the preprocessor collate format.

    Parameters
    ----------
    inference_input : InferenceInput
        Typed model input.

    Returns
    -------
    dict[str, numpy.typing.NDArray[Any]]
        Mapping accepted by the corresponding preprocessor's collate path.
    """
    if isinstance(inference_input, VoxelInferenceInput):
        output = {
            "voxel_features": inference_input.voxel_features,
            "voxel_coords": inference_input.voxel_coords,
        }
        if inference_input.voxel_num_points is not None:
            output["voxel_num_points"] = inference_input.voxel_num_points
        return output
    if isinstance(inference_input, BevInferenceInput):
        return {"bev_input": inference_input.bev_input}
    if isinstance(inference_input, PointCloudInferenceInput):
        return {"downsample_lidar": inference_input.points}
    raise TypeError(f"Unsupported inference input type: {type(inference_input).__name__}")


def merge_inference_inputs(inference_inputs: list[InferenceInput]) -> dict[str, list[npt.NDArray[Any]]]:
    """
    Merge homogeneous agent inputs for preprocessor collation.

    Parameters
    ----------
    inference_inputs : list[InferenceInput]
        Inputs for the agents participating in the current inference.

    Returns
    -------
    dict[str, list[numpy.typing.NDArray[Any]]]
        Dict-of-lists accepted by ``preprocessor.collate_batch()``.

    Raises
    ------
    ValueError
        If the input list is empty.
    TypeError
        If agents provide different input representations.
    """
    if not inference_inputs:
        raise ValueError("At least one inference input is required")

    input_type = type(inference_inputs[0])
    if any(type(inference_input) is not input_type for inference_input in inference_inputs[1:]):
        raise TypeError("All agents must use the same inference input representation")

    input_dicts = [inference_input_to_dict(inference_input) for inference_input in inference_inputs]
    input_fields = set(input_dicts[0])
    if any(set(input_dict) != input_fields for input_dict in input_dicts[1:]):
        raise TypeError("All agents must provide the same inference input fields")

    return {field: [input_dict[field] for input_dict in input_dicts] for field in input_dicts[0]}
