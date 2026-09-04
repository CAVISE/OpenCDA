"""Typed cooperative perception payloads exchanged over V2X."""

from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

from opencood.models.communication_adapters.inference_input import (
    DetectionInferenceInput,
    FpvrcnnAgentInferenceInput,
    MultiScaleFeatureInput,
    SpatialFeatureInput,
    Where2CommFeatureInput,
)
from opencood.models.communication_adapters.inference_metadata import (
    IntermediateMetadata,
    PoseFrameMetadata,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EarlyFusionWirePayload:
    """
    Sender-local LiDAR data exchanged for early fusion.

    Attributes
    ----------
    lidar_points : numpy.typing.NDArray[Any]
        LiDAR points expressed in the sender coordinate frame.
    metadata : PoseFrameMetadata
        Sender pose and capture frame used for receiver-side projection.
    """

    lidar_points: npt.NDArray[Any]
    metadata: PoseFrameMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class LateFusionWirePayload:
    """
    Sender-side detections exchanged for late fusion.

    Attributes
    ----------
    detections : DetectionInferenceInput
        Bounding-box candidates and confidence scores produced by the sender.
    metadata : PoseFrameMetadata
        Pose and capture frame required for late-fusion post-processing.
    """

    detections: DetectionInferenceInput
    metadata: PoseFrameMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class FpvrcnnWirePayload:
    """
    Agent-side FPV-RCNN output exchanged over V2X.

    Attributes
    ----------
    inference_input : FpvrcnnAgentInferenceInput
        Stage-one proposals and VSA keypoint features computed by the sender.
    metadata : PoseFrameMetadata
        Sender pose and capture frame used to transform the output at the
        receiver.
    """

    inference_input: FpvrcnnAgentInferenceInput
    metadata: PoseFrameMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class Where2CommWirePayload:
    """
    Sender-side learned features exchanged by Where2Comm.

    Attributes
    ----------
    inference_input : Where2CommFeatureInput
        Masked feature map and its effective communication rate.
    metadata : PoseFrameMetadata
        Sender pose and capture frame used by the fusion pipeline.
    """

    inference_input: Where2CommFeatureInput
    metadata: PoseFrameMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class IntermediateFeatureWirePayload:
    """
    Learned features exchanged by an intermediate-fusion model.

    Attributes
    ----------
    inference_input : SpatialFeatureInput | MultiScaleFeatureInput
        Model-specific learned features produced by the sender.
    metadata : IntermediateMetadata | None
        Metadata required by the receiver-side fusion implementation.
    """

    inference_input: SpatialFeatureInput | MultiScaleFeatureInput
    metadata: IntermediateMetadata | None
