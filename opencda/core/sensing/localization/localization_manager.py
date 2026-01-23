"""
Localization module for vehicle position and velocity estimation.

This module provides GNSS and IMU sensor management with Kalman filtering
for accurate vehicle localization in cooperative autonomous driving scenarios.
"""

import weakref
from collections import deque
from typing import Dict, Any, Optional

import carla
import numpy as np

from opencda.core.common.misc import get_speed
from opencda.core.sensing.localization.localization_debug_helper import LocDebugHelper
from opencda.core.sensing.localization.kalman_filter import KalmanFilter
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform


class GnssSensor(object):
    """
    GNSS sensor manager for vehicle localization.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The CARLA vehicle to attach the sensor to.
    config : Dict[str, Any]
        Configuration dictionary containing GNSS noise parameters.

    Attributes
    ----------
    sensor : carla.Sensor
        The GNSS sensor actor attached to the vehicle.
    lat : float
        Current latitude in degrees.
    lon : float
        Current longitude in degrees.
    alt : float
        Current altitude in meters.
    timestamp : float
        Timestamp of the latest GNSS measurement.
    """

    def __init__(self, vehicle: Any, config: Dict[str, Any]):
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.gnss")

        # set the noise for gps
        blueprint.set_attribute("noise_alt_stddev", str(config["noise_alt_stddev"]))
        blueprint.set_attribute("noise_lat_stddev", str(config["noise_lat_stddev"]))
        blueprint.set_attribute("noise_lon_stddev", str(config["noise_lon_stddev"]))
        # spawn the sensor
        self.sensor = world.spawn_actor(
            blueprint, carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)), attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid
        )

        # latitude and longitude at current timestamp
        self.lat, self.lon, self.alt, self.timestamp = 0.0, 0.0, 0.0, 0.0
        # create weak reference to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method that returns the current geo location."""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude
        self.timestamp = event.timestamp


class ImuSensor(object):
    """
    IMU sensor manager for vehicle motion sensing.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The CARLA vehicle to attach the sensor to.

    Attributes
    ----------
    sensor : carla.Sensor
        The IMU sensor actor attached to the vehicle.
    accelerometer : Tuple[float, float, float] or None
        3D acceleration measurements (x, y, z) in m/s².
    gyroscope : Tuple[float, float, float] or None
        3D angular velocity measurements (x, y, z) in rad/s.
    compass : float or None
        Compass heading in radians.
    """

    def __init__(self, vehicle: Any):
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: ImuSensor._IMU_callback(weak_self, sensor_data))
        self.gyroscope = None

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        """
        Callback for IMU measurement events.

        Parameters
        ----------
        weak_self : weakref.ref
            Weak reference to the ImuSensor instance.
        sensor_data : carla.IMUMeasurement
            IMU measurement event from CARLA.
        """
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        # m/s^2
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        )
        # rad/s
        self.gyroscope = (
            max(limits[0], min(limits[1], sensor_data.gyroscope.x)),
            max(limits[0], min(limits[1], sensor_data.gyroscope.y)),
            max(limits[0], min(limits[1], sensor_data.gyroscope.z)),
        )
        self.compass = sensor_data.compass


class LocalizationManager(object):
    """
    Localization module with sensor fusion for vehicle pose estimation.

    Combines GNSS and IMU measurements using Kalman filtering to estimate
    accurate vehicle position, heading, and velocity.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The CARLA vehicle for localization.
    config_yaml : Dict[str, Any]
        Configuration dictionary for localization parameters.
    carla_map : carla.Map
        CARLA HD map for coordinate transformations.

    Attributes
    ----------
    vehicle : carla.Vehicle
        The associated vehicle.
    activate : bool
        Whether sensor fusion is activated or using ground truth.
    map : carla.Map
        CARLA map instance.
    geo_ref : carla.GeoLocation
        Geographic reference point for coordinate conversion.
    gnss : GnssSensor
        GNSS sensor manager.
    imu : ImuSensor
        IMU sensor manager.
    kf : KalmanFilter
        Kalman filter for sensor fusion.
    debug_helper : LocDebugHelper
        Debug helper for visualization and evaluation.
    """

    def __init__(self, vehicle: Any, config_yaml: Dict[str, Any], carla_map: Any):
        self.vehicle = vehicle
        self.activate = config_yaml["activate"]
        self.map = carla_map
        self.geo_ref = self.map.transform_to_geolocation(carla.Location(x=0, y=0, z=0))

        # speed and transform and current timestamp
        self._ego_pos = None
        self._speed = 0

        # history track
        self._ego_pos_history = deque(maxlen=100)
        self._timestamp_history = deque(maxlen=100)

        self.gnss = GnssSensor(vehicle, config_yaml["gnss"])
        self.imu = ImuSensor(vehicle)

        # heading direction noise
        self.heading_noise_std = config_yaml["gnss"]["heading_direction_stddev"]
        self.speed_noise_std = config_yaml["gnss"]["speed_stddev"]

        self.dt = config_yaml["dt"]
        # Kalman Filter
        self.kf = KalmanFilter(self.dt)

        # DebugHelper
        self.debug_helper = LocDebugHelper(config_yaml["debug_helper"], self.vehicle.id)

    def localize(self):
        """
        Perform localization using sensor fusion or ground truth.

        Updates ego vehicle position and speed using GNSS, IMU, and Kalman
        filtering when activated, or retrieves ground truth from server when
        deactivated.
        """

        if not self.activate:
            self._ego_pos = self.vehicle.get_transform()
            self._speed = get_speed(self.vehicle)
        else:
            speed_true = get_speed(self.vehicle)
            speed_noise = self.add_speed_noise(speed_true)

            # gnss coordinates under ESU(Unreal coordinate system)
            x, y, z = geo_to_transform(self.gnss.lat, self.gnss.lon, self.gnss.alt, self.geo_ref.latitude, self.geo_ref.longitude, 0.0)

            # only use this for debugging purpose
            location = self.vehicle.get_transform().location

            # We add synthetic noise to the heading direction
            rotation = self.vehicle.get_transform().rotation
            heading_angle = self.add_heading_direction_noise(rotation.yaw)

            # assume the initial position is accurate
            if len(self._ego_pos_history) == 0:
                x_kf, y_kf, heading_angle_kf = x, y, heading_angle
                self._speed = speed_true
                self.kf.run_step_init(x, y, np.deg2rad(heading_angle), self._speed / 3.6)
            else:
                x_kf, y_kf, heading_angle_kf, speed_kf = self.kf.run_step(x, y, np.deg2rad(heading_angle), speed_noise / 3.6, self.imu.gyroscope[2])
                self._speed = speed_kf * 3.6
                heading_angle_kf = np.rad2deg(heading_angle_kf)

            # add data to debug helper
            self.debug_helper.run_step(
                x, y, heading_angle, speed_noise, x_kf, y_kf, heading_angle_kf, self._speed, location.x, location.y, rotation.yaw, speed_true
            )

            # the final pose of the vehicle
            self._ego_pos = carla.Transform(carla.Location(x=x_kf, y=y_kf, z=z), carla.Rotation(pitch=0, yaw=heading_angle_kf, roll=0))

            # save the track for future use
            self._ego_pos_history.append(self._ego_pos)
            self._timestamp_history.append(self.gnss.timestamp)

    def add_heading_direction_noise(self, heading_direction: float) -> float:
        """
        Add synthetic Gaussian noise to heading direction.

        Parameters
        ----------
        heading_direction : float
            Ground truth heading direction in degrees.

        Returns
        -------
        float
            Heading direction with added noise in degrees.
        """
        return heading_direction + np.random.normal(0, self.heading_noise_std)

    def add_speed_noise(self, speed: float) -> float:
        """
        Add Gaussian white noise to vehicle speed.

        Parameters
        ----------
        speed : float
            Ground truth speed in km/h.

        Returns
        -------
        float
            Speed with added noise in km/h.
        """
        return speed + np.random.normal(0, self.speed_noise_std)

    def get_ego_pos(self) -> Optional[carla.Transform]:
        """
        Get estimated ego vehicle position.

        Returns
        -------
        carla.Transform or None
            Estimated vehicle transform, or None if not initialized.
        """
        return self._ego_pos

    def get_ego_spd(self) -> float:
        """
        Get estimated ego vehicle speed.

        Returns
        -------
        float
            Estimated vehicle speed in km/h.
        """
        return self._speed

    def destroy(self):
        """
        Destroy the sensors
        """
        self.gnss.sensor.destroy()
        self.imu.sensor.destroy()
