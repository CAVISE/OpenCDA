"""Ground-truth localization provider."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import carla

from opencda.core.application.behavior.types import Transform
from opencda.core.sensing.localization.types import LocalizationSource, LocalizationState

if TYPE_CHECKING:
    from opencda.core.common.world_frame import WorldFrame


class GTLocalizer:
    """Read an entity's exact pose and velocity from its CARLA actor."""

    def __init__(self, actor: carla.Actor) -> None:
        self._actor = actor
        self._state: LocalizationState | None = None

    def update(self, world_frame: WorldFrame | None = None) -> LocalizationState:
        if world_frame is None:
            transform = self._actor.get_transform()
            velocity = self._actor.get_velocity()
            frame = None
            timestamp = None
        else:
            actor_state = world_frame.actor_state(self._actor.id)
            transform = actor_state.transform
            velocity = actor_state.velocity
            frame = world_frame.frame
            timestamp = world_frame.timestamp

        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self._state = LocalizationState(
            transform=Transform.from_carla(transform),
            speed_kmh=speed_kmh,
            source=LocalizationSource.GT,
            frame=frame,
            timestamp=timestamp,
        )
        return self._state

    def get_state(self) -> LocalizationState:
        if self._state is None:
            raise RuntimeError("GTLocalizer has no state. Call update() first.")
        return self._state

    def destroy(self) -> None:
        """Release localizer-owned resources; the CARLA actor is externally owned."""
