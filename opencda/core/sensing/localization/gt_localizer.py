"""Ground-truth localization provider."""

from __future__ import annotations

import math

import carla

from opencda.core.application.behavior.types import Transform
from opencda.core.sensing.localization.types import LocalizationSource, LocalizationState


class GTLocalizer:
    """Read an entity's exact pose and velocity from its CARLA actor."""

    def __init__(self, actor: carla.Actor) -> None:
        self._actor = actor
        self._state: LocalizationState | None = None

    def update(self) -> LocalizationState:
        velocity = self._actor.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self._state = LocalizationState(
            transform=Transform.from_carla(self._actor.get_transform()),
            speed_kmh=speed_kmh,
            source=LocalizationSource.GT,
        )
        return self._state

    def get_state(self) -> LocalizationState:
        if self._state is None:
            raise RuntimeError("GTLocalizer has no state. Call update() first.")
        return self._state

    def destroy(self) -> None:
        """Release localizer-owned resources; the CARLA actor is externally owned."""
