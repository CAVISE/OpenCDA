from dataclasses import dataclass
import carla


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

    @classmethod
    def from_carla(cls, other: carla.Location) -> "Location":
        return cls(x=other.x, y=other.y, z=other.z)

    def to_carla(self) -> carla.Location:
        return carla.Location(x=self.x, y=self.y, z=self.z)


@dataclass(frozen=True)
class Rotation:
    pitch: float = 0
    yaw: float = 0
    roll: float = 0

    @classmethod
    def from_carla(cls, other: carla.Rotation) -> "Rotation":
        return cls(pitch=other.pitch, yaw=other.yaw, roll=other.roll)

    def to_carla(self) -> carla.Rotation:
        return carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)


@dataclass(frozen=True)
class Transform:
    location: Location = Location()
    rotation: Rotation = Rotation()

    @classmethod
    def from_carla(cls, other: carla.Transform) -> "Transform":
        return cls(location=Location.from_carla(other.location), rotation=Rotation.from_carla(other.rotation))

    def to_carla(self) -> carla.Transform:
        return carla.Transform(self.location.to_carla(), self.rotation.to_carla())
