from __future__ import annotations
import carla
import math
from collections.abc import Sequence
from opencda.core.application.behavior.types import Location


def get_speed(vehicle: carla.Vehicle, meters: bool = False) -> float:
    """
    Compute speed of a vehicle in Km/h.

    Parameters
    ----------
    meters : bool
        Whether to use m/s (True) or km/h (False).

    vehicle : carla.vehicle
        The vehicle for which speed is calculated.

    Returns
    -------
    speed : float
        The vehicle speed.
    """
    vel = vehicle.get_velocity()
    vel_meter_per_second = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return vel_meter_per_second if meters else 3.6 * vel_meter_per_second


def calculate_target_speeds(
    trajectory: Sequence[Location],
    dt: float = 0.1,
    current_location: Location | carla.Location | None = None,
    current_speed: float | None = None,
    max_speed: float | None = None,
    max_accel: float | None = None,
    max_decel: float | None = None,
) -> list[float]:
    """
    Calculate per-point target speeds for a timed AIM trajectory.

    Parameters
    ----------
    trajectory : Sequence[Location]
        Predicted trajectory points in meters.

    dt : float
        Time delta between two predicted trajectory points, in seconds.
        The original AIM inference examples use 0.1 s.

    current_location : Location | carla.Location | None
        Current ego location. If provided, the speed for the first trajectory
        point is calculated from ego location to trajectory[0].

    current_speed : float | None
        Current ego speed in km/h. Used only for acceleration limiting.

    max_speed : float | None
        Optional speed cap in km/h.

    max_accel : float | None
        Optional acceleration cap in m/s^2.

    max_decel : float | None
        Optional deceleration cap in m/s^2. Pass a positive value.

    Returns
    -------
    list[float]
        Target speed for each trajectory point in km/h.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if not trajectory:
        return []

    speeds: list[float] = []
    previous_location = current_location
    previous_speed = current_speed

    for index, location in enumerate(trajectory):
        if previous_location is None:
            if len(trajectory) == 1:
                target_speed = current_speed if current_speed is not None else 0.0
            else:
                target_speed = _distance_2d(location, trajectory[min(index + 1, len(trajectory) - 1)]) / dt * 3.6
        else:
            target_speed = _distance_2d(previous_location, location) / dt * 3.6

        if max_speed is not None:
            target_speed = min(target_speed, max_speed)

        if previous_speed is not None:
            target_speed = _limit_speed_delta(
                target_speed=target_speed,
                previous_speed=previous_speed,
                dt=dt,
                max_accel=max_accel,
                max_decel=max_decel,
            )

        speeds.append(target_speed)
        previous_location = location
        previous_speed = target_speed

    return speeds


def _distance_2d(first: Location | carla.Location, second: Location | carla.Location) -> float:
    dx = second.x - first.x
    dy = second.y - first.y
    return (dx * dx + dy * dy) ** 0.5


def _limit_speed_delta(
    target_speed: float,
    previous_speed: float,
    dt: float,
    max_accel: float | None,
    max_decel: float | None,
) -> float:
    target_speed_mps = target_speed / 3.6
    previous_speed_mps = previous_speed / 3.6

    if max_accel is not None:
        target_speed_mps = min(target_speed_mps, previous_speed_mps + max_accel * dt)
    if max_decel is not None:
        target_speed_mps = max(target_speed_mps, previous_speed_mps - abs(max_decel) * dt)

    return max(0.0, target_speed_mps * 3.6)


def draw_trajetory_points(
    world: carla.World,
    locations: Sequence[Location],
    color: carla.Color = carla.Color(255, 0, 0),
    life_time: float = 5,
    size: float = 0.1,
) -> None:
    for location in locations:
        loc = carla.Location(location.x, location.y, location.z)
        world.debug.draw_point(loc, size=size, color=color, life_time=life_time)
