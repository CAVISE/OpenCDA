"""CARLA sensor adapters used by localization providers."""

from __future__ import annotations

import weakref
from typing import Any, Mapping

import carla


class GnssSensor:
    """Attach a CARLA GNSS sensor to an actor and retain its latest sample."""

    def __init__(self, actor: carla.Actor, config: Mapping[str, Any]) -> None:
        world = actor.get_world()
        blueprint = self.prepare_blueprint(world.get_blueprint_library(), config)
        sensor = world.spawn_actor(
            blueprint,
            self.spawn_transform(),
            attach_to=actor,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self._bind_sensor(sensor)

    @staticmethod
    def prepare_blueprint(blueprint_library: Any, config: Mapping[str, Any]) -> carla.ActorBlueprint:
        blueprint = blueprint_library.find("sensor.other.gnss")
        blueprint.set_attribute("noise_alt_stddev", str(config["noise_alt_stddev"]))
        blueprint.set_attribute("noise_lat_stddev", str(config["noise_lat_stddev"]))
        blueprint.set_attribute("noise_lon_stddev", str(config["noise_lon_stddev"]))
        return blueprint

    @staticmethod
    def spawn_transform() -> carla.Transform:
        return carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))

    @classmethod
    def from_sensor_actor(cls, sensor: carla.Actor) -> GnssSensor:
        instance = cls.__new__(cls)
        instance._bind_sensor(sensor)
        return instance

    def _bind_sensor(self, sensor: carla.Actor) -> None:
        self.sensor = sensor
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.timestamp = 0.0
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(
        weak_self: weakref.ReferenceType[GnssSensor],
        event: carla.GnssMeasurement,
    ) -> None:
        self = weak_self()
        if self is None:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude
        self.timestamp = event.timestamp


class ImuSensor:
    """Attach a CARLA IMU sensor to an actor and retain its latest sample."""

    def __init__(self, actor: carla.Actor) -> None:
        world = actor.get_world()
        sensor = world.spawn_actor(
            self.prepare_blueprint(world.get_blueprint_library()),
            self.spawn_transform(),
            attach_to=actor,
        )
        self._bind_sensor(sensor)

    @staticmethod
    def prepare_blueprint(blueprint_library: Any) -> carla.ActorBlueprint:
        return blueprint_library.find("sensor.other.imu")

    @staticmethod
    def spawn_transform() -> carla.Transform:
        return carla.Transform()

    @classmethod
    def from_sensor_actor(cls, sensor: carla.Actor) -> ImuSensor:
        instance = cls.__new__(cls)
        instance._bind_sensor(sensor)
        return instance

    def _bind_sensor(self, sensor: carla.Actor) -> None:
        self.sensor = sensor
        self.accelerometer: tuple[float, float, float] | None = None
        self.gyroscope: tuple[float, float, float] | None = None
        self.compass: float | None = None
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ImuSensor._on_imu_event(weak_self, event))

    @staticmethod
    def _on_imu_event(
        weak_self: weakref.ReferenceType[ImuSensor],
        event: carla.IMUMeasurement,
    ) -> None:
        self = weak_self()
        if self is None:
            return

        limits = (-99.9, 99.9)
        self.accelerometer = tuple(
            max(limits[0], min(limits[1], value)) for value in (event.accelerometer.x, event.accelerometer.y, event.accelerometer.z)
        )
        self.gyroscope = tuple(max(limits[0], min(limits[1], value)) for value in (event.gyroscope.x, event.gyroscope.y, event.gyroscope.z))
        self.compass = event.compass
