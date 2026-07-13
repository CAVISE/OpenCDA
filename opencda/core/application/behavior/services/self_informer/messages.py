"""Message dataclasses for the self-informer behavior service."""

from __future__ import annotations
from dataclasses import dataclass

from opencda.core.sensing.localization.types import LocalizationState


@dataclass(frozen=True)
class SelfInformerResponse:
    """Current localization state published by the self-informer service."""

    localization: LocalizationState
