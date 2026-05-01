from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    x: float = 0
    y: float = 0
    z: float = 0

    def __add__(self, other: "Location") -> "Location":
        if not isinstance(other, Location):
            return NotImplemented
        return Location(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: "Location") -> "Location":
        if not isinstance(other, Location):
            return NotImplemented
        return Location(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )


@dataclass(frozen=True)
class Rotation:
    pitch: float = 0
    yaw: float = 0
    roll: float = 0


@dataclass(frozen=True)
class Transform:
    location: Location = Location()
    rotation: Rotation = Rotation()
