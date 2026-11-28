"""Runtime state for an intercepted service method."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InterceptorBinding:
    """Track the dispatcher, original instance attribute and active rewrites.

    Attributes:
        dispatch: Wrapper installed on the service instance.
        overridden: Whether the instance had its own attribute before installation.
        previous: Original instance attribute to restore when overridden is true.
        callbacks: Registration tokens mapped to rewrites in installation order.
    """

    dispatch: Callable[..., Any]
    overridden: bool
    previous: Any
    callbacks: dict[object, Callable[[Any], Any]] = field(default_factory=dict)
