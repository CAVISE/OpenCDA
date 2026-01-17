# -*- coding: utf-8 -*-
"""
Coordinate system transformation utilities.

This module provides functions for transforming coordinates between different
coordinate systems, particularly WGS84 geodetic coordinates to East-North-Up
(ENU) local coordinate system.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import numpy as np
from typing import Tuple


def geo_to_transform(lat: float, lon: float, alt: float, lat_0: float, lon_0: float, alt_0: float) -> Tuple[float, float, float]:
    """
    Convert WGS84 geodetic coordinates to ENU local coordinates.

    The origin of the ENU (East-North-Up) coordinate system is defined by
    the geographic reference point. This function reverses the official
    transform_to_geo API.

    Parameters
    ----------
    lat : float
        Current latitude in degrees.
    lon : float
        Current longitude in degrees.
    alt : float
        Current altitude in meters.
    lat_0 : float
        Geographic reference latitude in degrees.
    lon_0 : float
        Geographic reference longitude in degrees.
    alt_0 : float
        Geographic reference altitude in meters.

    Returns
    -------
    x : float
        The transformed x coordinate (East) in meters.
    y : float
        The transformed y coordinate (North) in meters.
    z : float
        The transformed z coordinate (Up) in meters.
    """
    EARTH_RADIUS_EQUA = 6378137.0
    scale = np.cos(np.deg2rad(lat_0))

    mx = lon * np.pi * EARTH_RADIUS_EQUA * scale / 180
    mx_0 = scale * np.deg2rad(lon_0) * EARTH_RADIUS_EQUA
    x = mx - mx_0

    my = np.log(np.tan((lat + 90) * np.pi / 360)) * EARTH_RADIUS_EQUA * scale
    my_0 = scale * EARTH_RADIUS_EQUA * np.log(np.tan((90 + lat_0) * np.pi / 360))
    y = -(my - my_0)

    z = alt - alt_0

    return x, y, z
