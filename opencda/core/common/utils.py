from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla


def transform_to_tuple(transform: carla.Transform) -> tuple[float, float, float, float, float, float]:
    return (
        transform.location.x,
        transform.location.y,
        transform.location.z,
        transform.rotation.roll,
        transform.rotation.yaw,
        transform.rotation.pitch,
    )
