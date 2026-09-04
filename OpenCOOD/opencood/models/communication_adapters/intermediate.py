"""Reusable adapters for learned intermediate feature maps."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.inference_input import (
    InferenceInput,
    MultiScaleFeatureInput,
    SpatialFeatureInput,
    inference_input_to_dict,
)
from opencood.models.communication_adapters.inference_metadata import IntermediateMetadata
from opencood.models.communication_adapters.wire_payload import IntermediateFeatureWirePayload


@runtime_checkable
class IntermediateCommunicationDataset(Protocol):
    """Dataset capabilities required by learned-feature adapters."""

    pre_processor: Any

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[
            [InferenceInput, IntermediateMetadata | None],
            object,
        ],
    ) -> None:
        raise NotImplementedError


class IntermediateFeatureCommunicationAdapter(ModelCommunicationAdapter):
    """Base adapter for models exposing an ``encode_agent`` split."""

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """
        Encode and publish learned features for every local agent.

        Parameters
        ----------
        dataset : Any
            Intermediate dataset exposing individual agent inputs.
        idx : int
            Dataset frame to publish.
        """
        communication_dataset = self._require_dataset(dataset)
        communication_dataset.extract_data(
            idx,
            agent_payload_builder=lambda inference_input, metadata: self._build_wire_payload(
                communication_dataset,
                inference_input,
                metadata,
            ),
        )

    @staticmethod
    def _require_dataset(dataset: Any) -> IntermediateCommunicationDataset:
        if not isinstance(dataset, IntermediateCommunicationDataset):
            raise TypeError("Intermediate feature encoding requires an intermediate communication dataset")
        return dataset

    def _encode(
        self,
        dataset: IntermediateCommunicationDataset,
        inference_input: InferenceInput,
    ) -> Mapping[str, Any]:
        encode_agent = getattr(self.model, "encode_agent", None)
        if encode_agent is None:
            raise TypeError(f"{type(self).__name__} requires a model with encode_agent()")
        model_input = self.move_to_device(
            {
                "processed_lidar": dataset.pre_processor.collate_batch([inference_input_to_dict(inference_input)]),
                "batch_size": 1,
            }
        )
        self.model.eval()
        with torch.no_grad():
            output = encode_agent(model_input)
        if not isinstance(output, Mapping):
            raise TypeError("encode_agent() must return a tensor mapping")
        return output

    def _build_wire_payload(
        self,
        dataset: IntermediateCommunicationDataset,
        inference_input: InferenceInput,
        metadata: IntermediateMetadata | None,
    ) -> IntermediateFeatureWirePayload:
        encoded = self._encode(dataset, inference_input)
        return IntermediateFeatureWirePayload(
            inference_input=self._serialize_features(encoded),
            metadata=metadata,
        )

    def encode_local_intermediate_input(
        self,
        dataset: Any,
        inference_input: Any,
    ) -> dict[str, Any]:
        """
        Encode the receiver's own observation at the model boundary.

        Parameters
        ----------
        dataset : Any
            Intermediate dataset providing preprocessing.
        inference_input : Any
            Receiver-local typed preprocessor output.

        Returns
        -------
        dict[str, Any]
            Learned feature maps in the receiver assembly format.
        """
        communication_dataset = self._require_dataset(dataset)
        encoded = self._encode(communication_dataset, inference_input)
        return self._deserialize_features(self._serialize_features(encoded))

    def decode_received_payload(
        self,
        payload: object,
        receiver_pose: list[float],
    ) -> dict[str, Any]:
        """
        Decode one successfully delivered intermediate feature payload.

        Parameters
        ----------
        payload : object
            Deserialized model payload delivered over V2X.
        receiver_pose : list[float]
            Receiver pose reserved for model-side alignment.

        Returns
        -------
        dict[str, Any]
            Learned features and their sender metadata.
        """
        del receiver_pose
        if not isinstance(payload, IntermediateFeatureWirePayload):
            raise TypeError(f"Expected IntermediateFeatureWirePayload, got {type(payload).__name__}")
        decoded = self._deserialize_features(payload.inference_input)
        decoded["metadata"] = payload.metadata
        return decoded

    def _serialize_features(
        self,
        encoded: Mapping[str, Any],
    ) -> SpatialFeatureInput | MultiScaleFeatureInput:
        raise NotImplementedError

    def _deserialize_features(
        self,
        inference_input: SpatialFeatureInput | MultiScaleFeatureInput,
    ) -> dict[str, Any]:
        raise NotImplementedError


class SpatialFeatureCommunicationAdapter(IntermediateFeatureCommunicationAdapter):
    """Exchange one dense learned feature map per agent."""

    def _serialize_features(
        self,
        encoded: Mapping[str, Any],
    ) -> SpatialFeatureInput:
        """
        Move one encoded spatial feature map to the wire representation.

        Parameters
        ----------
        encoded : collections.abc.Mapping[str, Any]
            Model output containing ``spatial_features``.

        Returns
        -------
        SpatialFeatureInput
            CPU NumPy representation of the learned feature map.
        """
        spatial_features = encoded.get("spatial_features")
        if not isinstance(spatial_features, torch.Tensor):
            raise TypeError('encode_agent() field "spatial_features" must be a tensor')
        return SpatialFeatureInput(
            spatial_features=spatial_features.detach().cpu().numpy(),
        )

    def _deserialize_features(
        self,
        inference_input: SpatialFeatureInput | MultiScaleFeatureInput,
    ) -> dict[str, Any]:
        """
        Restore one spatial feature map for receiver-side collation.

        Parameters
        ----------
        inference_input : SpatialFeatureInput | MultiScaleFeatureInput
            Typed feature representation received over V2X.

        Returns
        -------
        dict[str, Any]
            Receiver assembly mapping containing ``spatial_features``.
        """
        if not isinstance(inference_input, SpatialFeatureInput):
            raise TypeError(f"Expected SpatialFeatureInput, got {type(inference_input).__name__}")
        return {
            "spatial_features": np.array(inference_input.spatial_features, copy=True),
        }


class MultiScaleFeatureCommunicationAdapter(IntermediateFeatureCommunicationAdapter):
    """Exchange an ordered tuple of learned feature scales per agent."""

    def _serialize_features(
        self,
        encoded: Mapping[str, Any],
    ) -> MultiScaleFeatureInput:
        """
        Move ordered model feature scales to the wire representation.

        Parameters
        ----------
        encoded : collections.abc.Mapping[str, Any]
            Model output containing ``feature_maps``.

        Returns
        -------
        MultiScaleFeatureInput
            CPU NumPy representations of all learned feature scales.
        """
        feature_maps = encoded.get("feature_maps")
        if not isinstance(feature_maps, (list, tuple)) or not all(isinstance(feature_map, torch.Tensor) for feature_map in feature_maps):
            raise TypeError('encode_agent() field "feature_maps" must be a tensor sequence')
        return MultiScaleFeatureInput(
            feature_maps=tuple(feature_map.detach().cpu().numpy() for feature_map in feature_maps),
        )

    def _deserialize_features(
        self,
        inference_input: SpatialFeatureInput | MultiScaleFeatureInput,
    ) -> dict[str, Any]:
        """
        Restore ordered feature scales for receiver-side collation.

        Parameters
        ----------
        inference_input : SpatialFeatureInput | MultiScaleFeatureInput
            Typed feature representation received over V2X.

        Returns
        -------
        dict[str, Any]
            Receiver assembly mapping keyed by scale index.
        """
        if not isinstance(inference_input, MultiScaleFeatureInput):
            raise TypeError(f"Expected MultiScaleFeatureInput, got {type(inference_input).__name__}")
        return {f"feature_{index}": np.array(feature_map, copy=True) for index, feature_map in enumerate(inference_input.feature_maps)}
