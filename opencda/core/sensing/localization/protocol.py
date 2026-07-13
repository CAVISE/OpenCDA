"""Localization provider contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from opencda.core.sensing.localization.types import LocalizationState


@runtime_checkable
class Localizer(Protocol):
    """Provide the latest localization state for an OpenCDA entity."""

    def update(self) -> LocalizationState:
        """Update and return the current localization state."""

    def get_state(self) -> LocalizationState:
        """Return the most recently produced localization state."""

    def destroy(self) -> None:
        """Release resources owned by the localizer."""
