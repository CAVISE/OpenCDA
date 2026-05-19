"""Capability definitions exposed by behavior services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, TypeAlias


class Capability(str, Enum):
    """Cross-service capability vocabulary."""

    REQUEST_OBSERVE = "request.observe"
    REQUEST_SUBMIT = "request.submit"

    RESPONSE_OBSERVE = "response.observe"
    RESPONSE_SUBMIT = "response.submit"

    COMMAND_SUBMIT = "command.submit"

    STATE_OBSERVE = "state.observe"


CapabilityBinding: TypeAlias = Callable[..., Any]
CapabilityBindings: TypeAlias = Mapping[Capability, CapabilityBinding]
