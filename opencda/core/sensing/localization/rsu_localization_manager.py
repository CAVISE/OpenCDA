# -*- coding: utf-8 -*-
"""
Localization module for roadside units (RSU).

This module provides GNSS-based localization functionality for infrastructure
units, including coordinate transformation from WGS84 to ENU coordinate system.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
from collections import deque
from typing import Dict, List, Any, Optional

import carla

from opencda.core.sensing.localization.coordinate_transform import geo_to_transform


class GnssSensor(object):
    """
    GNSS sensor module for roadside units.

    Parameters
    ----------
    world : carla.World
        CARLA world object.
    config : Dict[str, Any]
        Configuration dictionary for the GNSS sensor.
    global_position : List[float]
        Global position of the RSU as [x, y, z].

    Attributes
    ----------
    sensor : carla.Sensor
        GNSS sensor actor attached to the world.
    lat : float
        Current latitude in degrees.
    lon : float
        Current longitude in degrees.
    alt : float
        Current altitude in meters.
    timestamp : float
        Timestamp of the latest GNSS measurement.
    """

    def __init__(self, world: carla.World, config: Dict[str, Any], global_position: List[float]):
        blueprint = world.get_blueprint_library().find("sensor.other.gnss")

        # set the noise for gps
        blueprint.set_attribute("noise_alt_stddev", str(config["noise_alt_stddev"]))
        blueprint.set_attribute("noise_lat_stddev", str(config["noise_lat_stddev"]))
        blueprint.set_attribute("noise_lon_stddev", str(config["noise_lon_stddev"]))
        # spawn the sensor
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=global_position[0], y=global_position[1], z=global_position[2])))

        # latitude and longitude at current timestamp
        self.lat, self.lon, self.alt, self.timestamp = 0.0, 0.0, 0.0, 0.0
        # create weak reference to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self: weakref.ref, event: carla.GnssMeasurement) -> None:
        """
        Callback for GNSS measurement events.

        Parameters
        ----------
        weak_self : weakref.ref
            Weak reference to the GnssSensor instance.
        event : carla.GnssMeasurement
            GNSS measurement event from CARLA.
        """
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude
        self.timestamp = event.timestamp


class LocalizationManager(object):
    """
    Localization module for infrastructure units.

    Provides GNSS-based localization with coordinate transformation from
    WGS84 geodetic coordinates to ENU (East-North-Up) local coordinates.

    Parameters
    ----------
    world : carla.World
        CARLA world object.
    config_yaml : Dict[str, Any]
        Configuration dictionary for the localization module.
    carla_map : carla.Map
        CARLA HD map for coordinate system reference.

    Attributes
    ----------
    activate : bool
        Whether localization module is activated.
    map : carla.Map
        CARLA map object.
    geo_ref : carla.GeoLocation
        Geographic reference point (map origin).
    gnss : GnssSensor
        GNSS sensor manager.
    true_ego_pos : carla.Transform
        Ground truth position of the RSU.
    _ego_pos : carla.Transform or None
        Current estimated position.
    _speed : float
        Current speed (always 0 for RSU).
    _ego_pos_history : deque
        History of ego positions (max 100 entries).
    _timestamp_history : deque
        History of timestamps (max 100 entries).
    """

    def __init__(self, world: carla.World, config_yaml: Dict[str, Any], carla_map: carla.Map):
        self.activate = config_yaml["activate"]
        self.map = carla_map
        self.geo_ref = self.map.transform_to_geolocation(carla.Location(x=0, y=0, z=0))

        # speed and transform and current timestamp
        self._ego_pos = None
        self._speed = 0

        # history track
        self._ego_pos_history = deque(maxlen=100)
        self._timestamp_history = deque(maxlen=100)

        self.gnss = GnssSensor(world, config_yaml["gnss"], config_yaml["global_position"])
        self.true_ego_pos = carla.Transform(
            carla.Location(x=config_yaml["global_position"][0], y=config_yaml["global_position"][1], z=config_yaml["global_position"][2])
        )
        self._speed = 0

    def localize(self) -> None:
        """
        Perform localization using GNSS data or ground truth.

        If localization is deactivated, uses ground truth position.
        Otherwise, converts GNSS WGS84 coordinates to local ENU coordinates.
        """

        if not self.activate:
            self._ego_pos = self.true_ego_pos
        else:
            x, y, z = geo_to_transform(self.gnss.lat, self.gnss.lon, self.gnss.alt, self.geo_ref.latitude, self.geo_ref.longitude, 0.0)
            self._ego_pos = carla.Transform(carla.Location(x=x, y=y, z=z))

    def get_ego_pos(self) -> Optional[carla.Transform]:
        """
        Retrieve current ego position.

        Returns
        -------
        carla.Transform or None
            Current position transform, or None if not yet computed.
        """
        return self._ego_pos

    def get_ego_spd(self) -> float:
        """
        Retrieve ego speed.

        Returns
        -------
        float
            Current speed (always 0.0 for RSU).
        """
        return self._speed

    def destroy(self):
        """
        Destroy the sensors
        """
        self.gnss.sensor.destroy()
