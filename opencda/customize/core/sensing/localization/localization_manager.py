"""
Customized localization module for sensor fusion-based positioning.

This module extends the default OpenCDA localization manager with a customized
Extended Kalman Filter implementation for improved position estimation.
"""

from typing import Dict, Any
import carla

from opencda.core.sensing.localization.localization_manager import LocalizationManager
from opencda.customize.core.sensing.localization.extented_kalman_filter import ExtentedKalmanFilter


class CustomizedLocalizationManager(LocalizationManager):
    """
    Customized localization module with Extended Kalman Filter.

    Extends the default OpenCDA localization manager to use a custom
    Extended Kalman Filter for sensor fusion and position estimation.

    Parameters
    ----------
    vehicle : carla.Vehicle
        CARLA vehicle object for spawning GNSS and IMU sensors.
    config_yaml : Dict[str, Any]
        Configuration dictionary for the localization module.
    carla_map : carla.Map
        CARLA HD map for coordinate system conversion (WGS84 to ENU).

    Attributes
    ----------
    kf : ExtentedKalmanFilter
        Extended Kalman Filter for sensor fusion.
    dt : float
        Time step for filter updates (inherited from parent).
    vehicle : carla.Vehicle
        CARLA vehicle instance (inherited from parent).
    """

    def __init__(
        self,
        vehicle: carla.Vehicle,
        config_yaml: Dict[str, Any],
        carla_map: carla.Map,
    ):
        super(CustomizedLocalizationManager, self).__init__(vehicle, config_yaml, carla_map)
        self.kf = ExtentedKalmanFilter(self.dt) #NOTE Incompatible types in assignment (expression has type "ExtentedKalmanFilter", variable has type "KalmanFilter")
