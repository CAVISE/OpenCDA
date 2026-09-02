"""Typed cooperative perception payloads exchanged through Artery."""

from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

from opencood.models.communication_adapters.inference_input import (
    FpvrcnnAgentInferenceInput,
    InferenceInput,
)
from opencood.models.communication_adapters.inference_metadata import (
    IntermediateMetadata,
    PoseFrameMetadata,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EarlyFusionWirePayload:
    """Wire payload produced by :class:`EarlyFusionDataset`.

    Attributes
    ----------
    projected_lidar : numpy.typing.NDArray[Any]
        LiDAR points projected into the ego coordinate frame.
    """

    projected_lidar: npt.NDArray[Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class LateFusionWirePayload:
    """Wire payload produced by :class:`LateFusionDataset`.

    Attributes
    ----------
    inference_input : InferenceInput
        Typed model input produced by the sender's preprocessor.
    metadata : PoseFrameMetadata
        Pose and capture frame required for late-fusion post-processing.
    """

    inference_input: InferenceInput
    metadata: PoseFrameMetadata


@dataclass(frozen=True, slots=True, kw_only=True)
class IntermediateFusionWirePayload:
    """Wire payload produced by :class:`IntermediateFusionDataset`.

    Attributes
    ----------
    inference_input : InferenceInput
        Typed model input produced by the sender's preprocessor.
    metadata : IntermediateMetadata | None
        Metadata required by the configured model, or ``None`` when the model
        only consumes the inference input.
    """

    inference_input: InferenceInput
    metadata: IntermediateMetadata | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FpvrcnnWirePayload:
    """Agent-side FPV-RCNN output exchanged through Artery.

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
