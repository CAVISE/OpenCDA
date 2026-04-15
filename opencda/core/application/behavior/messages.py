"""Typed input and output messages for behavior applications."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorApplicationMessage:
    """Base transport-agnostic message passed into and out of behavior applications."""
