"""
V2X Position Falsification attack and detection.

Unlike GNSS Spoofing (which corrupts a vehicle's OWN position perception at
the sensor level), this attack falsifies the position/speed broadcast via
V2X (CAM messages) to deceive OTHER vehicles. The attacker knows its real
position but intentionally lies when other vehicles query it.
"""

import carla


class V2XConstantOffsetAttacker:
    """
    Reports position with a constant offset (e.g., shifted to adjacent lane).
    Speed is not modified.
    """

    def __init__(self, dx: float, dy: float, dz: float) -> None:
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def falsify_position(self, true_pos: carla.Transform) -> carla.Transform:
        loc = carla.Location(
            x=true_pos.location.x + self.dx,
            y=true_pos.location.y + self.dy,
            z=true_pos.location.z + self.dz,
        )
        return carla.Transform(loc, true_pos.rotation)

    def falsify_speed(self, true_speed: float) -> float:
        return true_speed


class V2XGhostVehicleAttacker:
    """
    Reports a completely fabricated ("ghost") location and speed,
    ignoring true values entirely.
    """

    def __init__(self, ghost_x: float, ghost_y: float, ghost_z: float, ghost_speed: float = 0.0) -> None:
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z
        self.ghost_speed = ghost_speed

    def falsify_position(self, true_pos: carla.Transform) -> carla.Transform:
        loc = carla.Location(x=self.ghost_x, y=self.ghost_y, z=self.ghost_z)
        return carla.Transform(loc, true_pos.rotation)

    def falsify_speed(self, true_speed: float) -> float:
        return self.ghost_speed


class V2XProgressiveDriftAttacker:
    """
    Gradually drifts reported position away from the real one by
    (dx, dy, dz) meters per tick. Speed is not modified.
    """

    def __init__(self, dx: float, dy: float, dz: float) -> None:
        self.tick = 0
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def falsify_position(self, true_pos: carla.Transform) -> carla.Transform:
        loc = carla.Location(
            x=true_pos.location.x + self.tick * self.dx,
            y=true_pos.location.y + self.tick * self.dy,
            z=true_pos.location.z + self.tick * self.dz,
        )
        self.tick += 1
        return carla.Transform(loc, true_pos.rotation)

    def falsify_speed(self, true_speed: float) -> float:
        return true_speed
