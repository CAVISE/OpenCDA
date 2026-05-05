"""Self-informer behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from .messages import SelfInformerResponse
from .utils import get_speed
from opencda.core.application.behavior.types import Location

if TYPE_CHECKING:
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.self_informer")


@BehaviorServiceRegistry.register
class SelfInformer:
    """Behavior service that publishes the owner's current CAV state."""

    service_type: str = "self_informer"
    priority: int = 1

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {
            Capability.REQUEST_SUBMIT: self._build_self_informer_response,
        }

    def __init__(self, priority: int = 1) -> None:
        """Initialize the self-informer behavior service."""
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None
        self.priority = priority

    def _get_owner(self) -> VehicleManager:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("Self-informer is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("Self-informer owner is no longer available.")

        return owner

    def on_attach(self, owner: VehicleManager) -> None:
        """Initialize the service for a particular participant instance."""
        self._owner_ref = weakref.ref(owner)

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None

    def get_state(self) -> Any:
        """Return an immutable snapshot of the current service state."""
        pass

    def _build_self_informer_response(
        self,
    ) -> TransportMessage[SelfInformerResponse]:
        owner = self._get_owner()
        current_location = Location.from_carla(owner.vehicle.get_location())
        current_speed = get_speed(owner.vehicle)
        payload = SelfInformerResponse(location=current_location, speed=current_speed)
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_type,
            dst_owner_id=owner.id,
            dst_service_type="broadcast",
            payload=payload,
        )

    def process(
        self,
        messages: Sequence[TransportMessage[Any]],
    ) -> tuple[TransportMessage[SelfInformerResponse], ...]:
        return (self._build_self_informer_response(),)
