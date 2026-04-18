from dataclasses import dataclass


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
