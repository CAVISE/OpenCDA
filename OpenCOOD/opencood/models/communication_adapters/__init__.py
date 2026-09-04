"""Model-owned communication adapter selection."""

from typing import cast

import torch
from torch import nn

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.early import EarlyFusionCommunicationAdapter
from opencood.models.communication_adapters.inference_input import (
    BevInferenceInput,
    DetectionInferenceInput,
    FpvrcnnAgentInferenceInput,
    InferenceInput,
    MultiScaleFeatureInput,
    PointCloudInferenceInput,
    SpatialFeatureInput,
    VoxelInferenceInput,
    Where2CommFeatureInput,
    build_inference_input,
    inference_input_to_dict,
    merge_inference_inputs,
)
from opencood.models.communication_adapters.intermediate import (
    IntermediateFeatureCommunicationAdapter,
    MultiScaleFeatureCommunicationAdapter,
    SpatialFeatureCommunicationAdapter,
)
from opencood.models.communication_adapters.inference_metadata import (
    IntermediateMetadata,
    PoseFrameMetadata,
    V2XViTMetadata,
)
from opencood.models.communication_adapters.late import LateFusionCommunicationAdapter
from opencood.models.communication_adapters.where2comm import Where2CommCommunicationAdapter
from opencood.models.communication_adapters.wire_payload import (
    EarlyFusionWirePayload,
    FpvrcnnWirePayload,
    IntermediateFeatureWirePayload,
    LateFusionWirePayload,
    Where2CommWirePayload,
)


def build_communication_adapter(
    model: nn.Module,
    device: torch.device,
    fusion_method: str | None = None,
) -> ModelCommunicationAdapter:
    """
    Build the communication adapter declared by a perception model.

    Fusion-wide contracts are selected from the configured fusion method.
    Intermediate models declare the adapter matching their learned-feature
    communication boundary.

    Parameters
    ----------
    model : torch.nn.Module
        Cooperative perception model instance.
    device : torch.device
        Device used for sender-side model execution.
    fusion_method : str | None
        Explicit fusion method selected by the model configuration.

    Returns
    -------
    ModelCommunicationAdapter
        Adapter responsible for the model's network communication boundary.

    Raises
    ------
    TypeError
        If the model declares an invalid adapter class.
    """
    if fusion_method == "EarlyFusionDataset":
        return EarlyFusionCommunicationAdapter(model, device)
    if fusion_method == "LateFusionDataset":
        return LateFusionCommunicationAdapter(model, device)

    adapter_class: object = getattr(type(model), "communication_adapter_class", None)
    if adapter_class is None:
        raise TypeError(f"{type(model).__name__} must declare communication_adapter_class")
    if not isinstance(adapter_class, type) or not issubclass(
        adapter_class,
        ModelCommunicationAdapter,
    ):
        raise TypeError("model.communication_adapter_class must derive from ModelCommunicationAdapter")
    return cast(type[ModelCommunicationAdapter], adapter_class)(model, device)


__all__ = (
    "BevInferenceInput",
    "DetectionInferenceInput",
    "EarlyFusionWirePayload",
    "EarlyFusionCommunicationAdapter",
    "FpvrcnnAgentInferenceInput",
    "FpvrcnnWirePayload",
    "InferenceInput",
    "IntermediateFeatureCommunicationAdapter",
    "IntermediateFeatureWirePayload",
    "IntermediateMetadata",
    "LateFusionWirePayload",
    "LateFusionCommunicationAdapter",
    "ModelCommunicationAdapter",
    "MultiScaleFeatureCommunicationAdapter",
    "MultiScaleFeatureInput",
    "PointCloudInferenceInput",
    "PoseFrameMetadata",
    "SpatialFeatureCommunicationAdapter",
    "SpatialFeatureInput",
    "V2XViTMetadata",
    "VoxelInferenceInput",
    "Where2CommCommunicationAdapter",
    "Where2CommFeatureInput",
    "Where2CommWirePayload",
    "build_communication_adapter",
    "build_inference_input",
    "inference_input_to_dict",
    "merge_inference_inputs",
)
