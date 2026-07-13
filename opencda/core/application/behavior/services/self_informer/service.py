"""Self-informer behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage, BROADCAST_SERVICE_TYPE
from .messages import SelfInformerResponse
from .types import SelfInformerState

if TYPE_CHECKING:
    from opencda.core.common.rsu_manager import RSUManager
    from opencda.core.common.vehicle_manager import VehicleManager
    from opencda.core.sensing.localization.types import LocalizationState


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
        self._owner_ref: weakref.ReferenceType[VehicleManager | RSUManager] | None = None
        self.priority = priority
        self.localization: LocalizationState | None = None

    def _get_owner(self) -> VehicleManager | RSUManager:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("Self-informer is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("Self-informer owner is no longer available.")

        return owner

    def on_attach(self, owner: VehicleManager | RSUManager) -> None:
        """Initialize the service for a particular participant instance."""
        self._owner_ref = weakref.ref(owner)
        self.localization = None

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self.localization = None

    def get_state(self) -> SelfInformerState:
        """Return an immutable snapshot of the current service state."""
        owner = self._get_owner()
        if self.localization is None:
            self.localization = owner.localizer.get_state()
        return SelfInformerState(
            service_type=self.service_type,
            owner_id=owner.id,
            localization=self.localization,
        )

    def _build_self_informer_response(
        self,
    ) -> TransportMessage[SelfInformerResponse]:
        owner = self._get_owner()
        self.localization = owner.localizer.get_state()
        payload = SelfInformerResponse(localization=self.localization)
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_type,
            dst_owner_id=owner.id,
            dst_service_type=BROADCAST_SERVICE_TYPE,
            payload=payload,
        )

    def process(
        self,
        messages: Sequence[TransportMessage[Any]],
    ) -> tuple[TransportMessage[SelfInformerResponse], ...]:
        return (self._build_self_informer_response(),)
