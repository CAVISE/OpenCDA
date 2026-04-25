"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from typing import Any, Sequence, cast, TYPE_CHECKING

from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage

from AIM import get_model

if TYPE_CHECKING:
    from opencda.core.common.rsu_manager import RSUManager

    from .messages import AIMServerRequest, AIMServerResponse
from .aim_model_manager import AIMModelManager

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server")


@BehaviorServiceRegistry.register
class AIMServer:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name = "aim_server"
    priority = 20

    def __init__(
        self,
        priority: int = 20,
        **aim_config: Any,
    ) -> None:
        """
        Initialize the AIM-backed behavior service.

        Parameters
        ----------
        model : AIMModel
            Loaded AIM model used for trajectory prediction.
        """
        self._owner_ref: weakref.ReferenceType[RSUManager] | None = None
        self.aim_model_manager: AIMModelManager | None = None
        self.priority = priority

        aim_model_name = cast(str, aim_config.pop("model", "MTP"))
        self.model = get_model(aim_model_name, **aim_config)

    def _get_owner(self) -> RSUManager:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("AIM server is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("AIM server owner is no longer available.")

        return owner

    def on_attach(self, owner: RSUManager) -> None:
        """Initialize the service for a particular participant instance."""
        self._owner_ref = weakref.ref(owner)

        owner_instance = self._get_owner()
        owner_instance.localizer.localize()
        control_center = owner_instance.localizer.get_ego_pos()
        if control_center is None:
            raise RuntimeError("AIM server could not resolve the node localization control center.")
        self.aim_model_manager = AIMModelManager(self.model, control_center, self.service_name, owner_instance.id)

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self.aim_model_manager = None

    def process(self, messages: Sequence[TransportMessage[AIMServerRequest]]) -> Sequence[TransportMessage[AIMServerResponse]]:
        aim_model_manager = self.aim_model_manager
        if aim_model_manager is None:
            raise RuntimeError("AIM server is not attached to an owner.")

        result = aim_model_manager.process(messages)
        return result
