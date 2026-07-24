"""MapManager operating modes."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping


class MapManagerMode(StrEnum):
    """Supported levels of map processing."""

    DISABLED = "disabled"
    OFFROAD_ONLY = "offroad_only"
    FULL_BEV = "full_bev"


def resolve_map_manager_mode(config: Mapping[str, Any]) -> MapManagerMode:
    """Resolve the configured mode, including the legacy activate flag."""
    if "activate" in config:
        activate = config["activate"]
        if not isinstance(activate, bool):
            raise TypeError("MapManager config key 'activate' must be a boolean.")
        return MapManagerMode.FULL_BEV if activate else MapManagerMode.DISABLED

    mode = config.get("mode", MapManagerMode.OFFROAD_ONLY.value)
    if not isinstance(mode, str):
        raise TypeError("MapManager config key 'mode' must be a string.")
    try:
        return MapManagerMode(mode)
    except ValueError as exc:
        supported = ", ".join(item.value for item in MapManagerMode)
        raise ValueError(f"Unsupported MapManager mode {mode!r}. Expected one of: {supported}.") from exc
