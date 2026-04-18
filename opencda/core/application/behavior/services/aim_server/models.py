from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Location:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Rotation:
    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class Transform:
    location: Location
    rotation: Rotation


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
