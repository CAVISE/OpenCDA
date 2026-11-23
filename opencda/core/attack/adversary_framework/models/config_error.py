"""Location of an invalid value inside an attack definition."""

from collections.abc import Iterator
from contextlib import contextmanager


class ConfigError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}" if path else reason)


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
