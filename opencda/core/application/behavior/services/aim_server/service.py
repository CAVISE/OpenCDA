"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from typing import Any, Sequence, cast, TYPE_CHECKING, Mapping

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage

from AIM import get_model

if TYPE_CHECKING:
    from opencda.core.common.rsu_manager import RSUManager
    from .messages import AIMServerRequest, AIMServerResponse
    from opencda.core.application.behavior.types import Location

from .aim_model_manager import AIMModelManager
from .types import AIMServerState
from .utils import parse_location

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server")


@BehaviorServiceRegistry.register
class AIMServer:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_type = "aim_server"
    priority = 20

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {
            Capability.REQUEST_OBSERVE: self._observe_aim_requests,
            Capability.RESPONSE_SUBMIT: self._build_aim_response_messages,
            Capability.STATE_OBSERVE: self.get_state,
        }

    def __init__(
        self,
        priority: int = 20,
        control_radius: int = 15,
        control_center_location: Location | Mapping | Sequence | None = None,
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

        self.control_radius: int = control_radius
        self.control_center_location: Location | None = parse_location(control_center_location)
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
        if self.control_center_location is None:
            owner_instance.localizer.localize()
            control_center = owner_instance.localizer.get_ego_pos()
            if control_center is None:
                raise RuntimeError("AIM server could not resolve the node localization control center.")
            self.control_center_location = control_center.location
        self.aim_model_manager = AIMModelManager(self.model, self.control_center_location, self.service_type, owner_instance.id, self.control_radius)

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self.aim_model_manager = None

    def get_state(self) -> AIMServerState | None:
        if self.aim_model_manager is None:
            return AIMServerState(
                service_type=self.service_type,
                owner_id=None,
                is_attached=False,
                tracked_vehicle_ids=(),
                trajectory_vehicle_ids=(),
                tracked_vehicle_count=0,
                trajectory_vehicle_count=0,
            )

        return self.aim_model_manager.get_state_snapshot()

    def _observe_aim_requests(
        self,
        messages: Sequence[TransportMessage[AIMServerRequest]],
    ) -> tuple[TransportMessage[AIMServerRequest], ...]:
        return tuple(messages)

    def _build_aim_response_messages(
        self,
        messages: Sequence[TransportMessage[AIMServerRequest]],
    ) -> tuple[TransportMessage[AIMServerResponse], ...]:
        aim_model_manager = self.aim_model_manager
        if aim_model_manager is None:
            raise RuntimeError("AIM server is not attached to an owner.")

        return aim_model_manager.process(messages)

    def process(self, messages: Sequence[TransportMessage[AIMServerRequest]]) -> tuple[TransportMessage[AIMServerResponse], ...]:
        observed_requests = self._observe_aim_requests(messages)
        return self._build_aim_response_messages(observed_requests)
