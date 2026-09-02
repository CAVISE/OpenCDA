"""Temporary adapters for models awaiting a true network boundary."""

from typing import Any

from opencood.models.communication_adapters.base import ModelCommunicationAdapter


class PendingModelCommunicationAdapter(ModelCommunicationAdapter):
    """Preserve the current dataset-managed payload until a model is split.

    This adapter intentionally performs no model-side encoding. It exists so
    models that have not been migrated yet still use their current payload
    implementation without adding model-name branches to the manager.

    TODO
    ----
    Replace this fallback with model-side adapters for Early, Late and the
    remaining Intermediate models. Each adapter must place the wire boundary
    at the model's real communication stage instead of sending preprocessor
    output by default.
    """

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """Delegate payload construction to the existing dataset path.

        Parameters
        ----------
        dataset : Any
            Dataset retaining its temporary payload implementation.
        idx : int
            Dataset frame to publish.
        """
        dataset.extract_data(idx)
