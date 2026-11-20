"""Resolve scenario map names against the connected CARLA server."""

from collections.abc import Sequence
from pathlib import PurePosixPath


def resolve_carla_map(town: str, available_maps: Sequence[str]) -> str:
    """Keep exact names; allow classic towns to use CARLA 1.0's layered maps."""
    if town in available_maps:
        return town
    names = [town]
    if town in {f"Town{number:02d}" for number in range(1, 8)}:
        names.append(f"{town}_Opt")
    for name in names:
        matches = [path for path in available_maps if PurePosixPath(path).name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Map {name!r} is ambiguous; configure its full package path: {matches}")
    raise RuntimeError(f"Map {town!r} is unavailable. Install the matching AdditionalMaps pack. Available maps: {list(available_maps)}")
