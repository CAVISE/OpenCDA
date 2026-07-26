"""MapManager operating modes."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TypedDict


class MapManagerMode(StrEnum):
    """Supported levels of map processing."""

    DISABLED = "disabled"
    OFFROAD_ONLY = "offroad_only"
    FULL_BEV = "full_bev"


class MapManagerConfig(TypedDict, total=False):
    """Supported MapManager configuration values."""

    mode: str
    activate: bool
    visualize: bool
    pixels_per_meter: float
    raster_size: Sequence[float]
    lane_sample_resolution: float


def resolve_map_manager_mode(config: MapManagerConfig) -> MapManagerMode:
    """Resolve the configured map-processing mode.

    Parameters
    ----------
    config : MapManagerConfig
        MapManager configuration containing ``mode`` or the legacy
        ``activate`` flag.

    Returns
    -------
    MapManagerMode
        Validated MapManager operating mode.

    Raises
    ------
    TypeError
        If ``mode`` or ``activate`` has an invalid type.
    ValueError
        If the configured mode is unsupported.
    """
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
