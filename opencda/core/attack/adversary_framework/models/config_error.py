"""Location of an invalid value inside an attack definition."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


class ConfigError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}" if path else reason)


def check_config_keys(data: Mapping[str, Any], allowed: set[str], *, prefix: str = "") -> None:
    """Report an unexpected key relative to the current configuration path."""
    for key in data:
        if key not in allowed:
            raise ConfigError(f"{prefix}{key}", "Unknown field.")


@contextmanager
def at_config_path(path: str) -> Iterator[None]:
    """Keep the complete YAML path as nested parsers report local errors."""
    try:
        yield
    except ConfigError as exc:
        separator = "." if exc.path and not exc.path.startswith("[") else ""
        raise ConfigError(f"{path}{separator}{exc.path}", exc.reason) from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(path, str(exc)) from exc
