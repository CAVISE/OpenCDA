from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True)
class CavData:
    intention: str
    pos: Location
    sumo_pos: np.ndarray
    speed: float
    yaw: float
    waypoints: Sequence[Any]

    src_owner_id: str
    src_service_type: str
    dst_owner_id: str
    dst_service_type: str


@dataclass(frozen=True, slots=True)
class AIMServerState:
    service_name: str
    owner_id: str | None
    is_attached: bool
    tracked_vehicle_ids: tuple[str, ...]
    trajectory_vehicle_ids: tuple[str, ...]
    tracked_vehicle_count: int
    trajectory_vehicle_count: int
