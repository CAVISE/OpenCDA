from dataclasses import dataclass
import numpy as np
from opencda.core.application.behavior.types import Location


@dataclass(frozen=True)
class CavData:
    intention: str
    pos: Location
    sumo_pos: np.ndarray
    speed: float
    yaw: float
    waypoints: list

    src_owner_id: str
    src_service_type: str
    dst_owner_id: str
    dst_service_type: str
