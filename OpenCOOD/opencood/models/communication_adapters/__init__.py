"""Model-owned communication adapter selection."""

from typing import cast

import torch
from torch import nn

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.inference_input import (
    BevInferenceInput,
    FpvrcnnAgentInferenceInput,
    InferenceInput,
    PointCloudInferenceInput,
    VoxelInferenceInput,
    build_inference_input,
    inference_input_to_dict,
    merge_inference_inputs,
)
from opencood.models.communication_adapters.inference_metadata import (
    IntermediateMetadata,
    PoseFrameMetadata,
    V2XViTMetadata,
)
from opencood.models.communication_adapters.pending import PendingModelCommunicationAdapter
from opencood.models.communication_adapters.wire_payload import (
    EarlyFusionWirePayload,
    FpvrcnnWirePayload,
    IntermediateFusionWirePayload,
    LateFusionWirePayload,
)


def build_communication_adapter(
    model: nn.Module,
    device: torch.device,
) -> ModelCommunicationAdapter:
    """
    Build the communication adapter declared by a perception model.

    Models without a declared adapter temporarily retain their dataset-managed
    payload through :class:`PendingModelCommunicationAdapter`.

    Parameters
    ----------
    model : torch.nn.Module
        Cooperative perception model instance.
    device : torch.device
        Device used for sender-side model execution.

    Returns
    -------
    ModelCommunicationAdapter
        Adapter responsible for the model's network communication boundary.

    Raises
    ------
    TypeError
        If the model declares an invalid adapter class.
    """
    # TODO(#191): Remove the fallback after every model declares the adapter
    # matching its real sender/receiver communication boundary.
    adapter_class: object = getattr(
        type(model),
        "communication_adapter_class",
        PendingModelCommunicationAdapter,
    )
    if not isinstance(adapter_class, type) or not issubclass(
        adapter_class,
        ModelCommunicationAdapter,
    ):
        raise TypeError("model.communication_adapter_class must derive from ModelCommunicationAdapter")
    return cast(type[ModelCommunicationAdapter], adapter_class)(model, device)


__all__ = (
    "BevInferenceInput",
    "EarlyFusionWirePayload",
    "FpvrcnnAgentInferenceInput",
    "FpvrcnnWirePayload",
    "InferenceInput",
    "IntermediateFusionWirePayload",
    "IntermediateMetadata",
    "LateFusionWirePayload",
    "ModelCommunicationAdapter",
    "PendingModelCommunicationAdapter",
    "PointCloudInferenceInput",
    "PoseFrameMetadata",
    "V2XViTMetadata",
    "VoxelInferenceInput",
    "build_communication_adapter",
    "build_inference_input",
    "inference_input_to_dict",
    "merge_inference_inputs",
)
