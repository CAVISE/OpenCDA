"""Typed processing results produced by behavior applications."""

from dataclasses import dataclass, field

from .messages import BehaviorApplicationMessage


@dataclass(frozen=True)
class BehaviorApplicationResult:
    """Typed wrapper around the result produced by a single behavior application."""

    application_id: str
    messages: tuple[BehaviorApplicationMessage, ...] = field(default_factory=tuple)
