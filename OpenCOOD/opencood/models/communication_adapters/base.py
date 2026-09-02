"""Common interface for model-specific network communication adapters."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ModelCommunicationAdapter(ABC):
    """Bridge a perception model to its sender/receiver wire contract."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @abstractmethod
    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """Build and publish every sender payload for one dataset frame.

        Parameters
        ----------
        dataset : Any
            OpenCOOD dataset providing the local scene observations.
        idx : int
            Dataset frame to encode and publish.
        """

    def move_to_device(self, value: Any) -> Any:
        """Recursively move a model input to the adapter device.

        Parameters
        ----------
        value : Any
            Nested tensors, mappings, or sequences forming a model input.

        Returns
        -------
        Any
            Input with every tensor moved to the configured device.
        """
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self.move_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.move_to_device(item) for item in value)
        return value

    def decode_received_payload(
        self,
        payload: object,
        receiver_pose: list[float],
    ) -> dict[str, Any]:
        """Decode one delivered model payload for receiver-side inference.

        Parameters
        ----------
        payload : object
            Deserialized model payload delivered by the communication layer.
        receiver_pose : list[float]
            Receiver pose used for model-specific coordinate conversion.

        Returns
        -------
        dict[str, Any]
            Model-specific remote inference input.

        Raises
        ------
        NotImplementedError
            If the model still uses dataset-managed payload decoding.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement receiver-side payload decoding")
