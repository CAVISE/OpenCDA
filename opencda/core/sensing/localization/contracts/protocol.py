"""Localization provider contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from opencda.core.sensing.localization.contracts.types import LocalizationState

if TYPE_CHECKING:
    from opencda.core.common.world_frame import WorldFrame


@runtime_checkable
class Localizer(Protocol):
    """Provide the latest localization state for an OpenCDA entity."""

    def update(self, world_frame: WorldFrame | None = None) -> LocalizationState:
        """Update and return the current localization state."""

    def get_state(self) -> LocalizationState:
        """Return the most recently produced localization state."""

    def destroy(self) -> None:
        """Release resources owned by the localizer."""
