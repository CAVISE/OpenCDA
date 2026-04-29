"""Runtime context shared across attack stages and service interceptors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService

if TYPE_CHECKING:
    from ..attack import Attack


@dataclass(slots=True)
class AttackContext:
    """Mutable execution context for a running attack."""

    attack: Attack
    available_services: tuple[BehaviorService[Any, Any], ...]
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _active_interceptors: dict[str, Callable[[], None]] = field(default_factory=dict, repr=False)
