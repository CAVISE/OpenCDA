"""Where2Comm intermediate-feature communication adapter."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.inference_input import (
    InferenceInput,
    Where2CommFeatureInput,
    inference_input_to_dict,
)
from opencood.models.communication_adapters.inference_metadata import PoseFrameMetadata
from opencood.models.communication_adapters.wire_payload import Where2CommWirePayload


@runtime_checkable
class Where2CommCommunicationDataset(Protocol):
    """Dataset capabilities required by the Where2Comm adapter."""

    pre_processor: Any

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[
            [InferenceInput, PoseFrameMetadata],
            object,
        ],
    ) -> None:
        raise NotImplementedError


class Where2CommCommunicationAdapter(ModelCommunicationAdapter):
    """Execute the private Where2Comm encoder before V2X transmission."""

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """
        Encode and publish learned features for every local agent.

        Parameters
        ----------
        dataset : Any
            Dataset exposing individual intermediate-fusion inputs.
        idx : int
            Dataset frame to publish.

        Raises
        ------
        TypeError
            If the selected dataset lacks the required capabilities.
        """
        if not isinstance(dataset, Where2CommCommunicationDataset):
            raise TypeError("Where2Comm requires an intermediate-fusion communication dataset")
        dataset.extract_data(
            idx,
            agent_payload_builder=lambda inference_input, metadata: self._build_wire_payload(
                dataset,
                inference_input,
                metadata,
            ),
        )

    def _encode(
        self,
        dataset: Where2CommCommunicationDataset,
        inference_input: InferenceInput,
        *,
        apply_communication_mask: bool,
    ) -> tuple[np.ndarray, float]:
        """
        Run the sender-private model stages for one agent.

        Parameters
        ----------
        dataset : Where2CommCommunicationDataset
            Dataset providing the configured preprocessor.
        inference_input : InferenceInput
            One agent's preprocessed LiDAR input.
        apply_communication_mask : bool
            Whether to apply the over-the-air feature selection mask.

        Returns
        -------
        tuple[numpy.ndarray, float]
            Learned feature map and selected-location ratio.

        Raises
        ------
        TypeError
            If the model does not expose the Where2Comm encoder contract.
        """
        encode_agent = getattr(self.model, "encode_agent", None)
        if encode_agent is None:
            raise TypeError("Where2CommCommunicationAdapter requires a model with encode_agent()")

        model_input = self.move_to_device(
            {
                "processed_lidar": dataset.pre_processor.collate_batch([inference_input_to_dict(inference_input)]),
            }
        )
        self.model.eval()
        with torch.no_grad():
            output = encode_agent(
                model_input,
                apply_communication_mask=apply_communication_mask,
            )
        if not isinstance(output, Mapping):
            raise TypeError("Where2Comm encode_agent() must return a tensor mapping")
        spatial_features = output.get("spatial_features")
        communication_rate = output.get("communication_rate")
        if not isinstance(spatial_features, torch.Tensor):
            raise TypeError('Where2Comm output field "spatial_features" must be a tensor')
        if not isinstance(communication_rate, torch.Tensor):
            raise TypeError('Where2Comm output field "communication_rate" must be a tensor')
        return (
            spatial_features.detach().cpu().numpy(),
            float(communication_rate.detach().cpu().item()),
        )

    def _build_wire_payload(
        self,
        dataset: Where2CommCommunicationDataset,
        inference_input: InferenceInput,
        metadata: PoseFrameMetadata,
    ) -> Where2CommWirePayload:
        """
        Build one sender's learned-feature payload.

        Parameters
        ----------
        dataset : Where2CommCommunicationDataset
            Dataset providing the configured preprocessor.
        inference_input : InferenceInput
            Sender-local preprocessor output.
        metadata : PoseFrameMetadata
            Sender pose and capture frame.

        Returns
        -------
        Where2CommWirePayload
            Masked learned features and sender metadata.
        """
        spatial_features, communication_rate = self._encode(
            dataset,
            inference_input,
            apply_communication_mask=True,
        )
        return Where2CommWirePayload(
            inference_input=Where2CommFeatureInput(
                spatial_features=spatial_features,
                communication_rate=communication_rate,
            ),
            metadata=metadata,
        )

    def encode_local_intermediate_input(
        self,
        dataset: Any,
        inference_input: Any,
    ) -> dict[str, Any]:
        """
        Encode the receiver's own feature without an over-the-air mask.

        Parameters
        ----------
        dataset : Any
            Intermediate-fusion dataset providing the preprocessor.
        inference_input : Any
            Receiver-local typed preprocessor output.

        Returns
        -------
        dict[str, Any]
            Receiver-local learned feature input.
        """
        if not isinstance(dataset, Where2CommCommunicationDataset):
            raise TypeError("Where2Comm requires an intermediate-fusion communication dataset")
        spatial_features, communication_rate = self._encode(
            dataset,
            inference_input,
            apply_communication_mask=False,
        )
        return {
            "spatial_features": spatial_features,
            "communication_rate": communication_rate,
        }

    def decode_received_payload(
        self,
        payload: object,
        receiver_pose: list[float],
    ) -> dict[str, Any]:
        """
        Decode one successfully delivered Where2Comm feature payload.

        Parameters
        ----------
        payload : object
            Deserialized Where2Comm payload delivered over V2X.
        receiver_pose : list[float]
            Receiver pose reserved for later receiver-side feature alignment.

        Returns
        -------
        dict[str, Any]
            Learned feature map and communication rate.

        Raises
        ------
        TypeError
            If a payload belonging to another contract is received.
        """
        del receiver_pose
        if not isinstance(payload, Where2CommWirePayload):
            raise TypeError(f"Expected Where2CommWirePayload, got {type(payload).__name__}")
        return {
            "spatial_features": np.array(
                payload.inference_input.spatial_features,
                copy=True,
            ),
            "communication_rate": payload.inference_input.communication_rate,
            "metadata": payload.metadata,
        }
