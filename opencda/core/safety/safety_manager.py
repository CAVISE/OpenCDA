"""
Safety manager for monitoring autonomous vehicle hazards.

This module provides safety monitoring functionality for autonomous vehicles,
including collision detection, traffic light violations, stuck detection, and
off-road detection with optional human takeover capabilities.
"""

import logging
from typing import Dict, Any, Protocol, List
import carla

from opencda.core.safety.sensors import CollisionSensor, TrafficLightDector, StuckDetector, OffRoadDetector

logger = logging.getLogger("cavise.safety_manager")


class SafetyManager:
    """
    Safety monitoring manager for autonomous vehicles.

    Manages multiple safety sensors and monitors hazard status for a vehicle
    in the simulation environment, providing warnings and control handover
    capabilities when necessary.

    Parameters
    ----------
    vehicle : carla.Actor
        The vehicle that the SafetyManager is responsible for monitoring.
    params : Dict[str, Any]
        Configuration dictionary containing parameters for safety sensors:

        - print_message : bool
            Whether to print safety warnings.
        - collision_sensor : dict
            Configuration for collision sensor.
        - stuck_dector : dict
            Configuration for stuck detector.
        - offroad_dector : dict
            Configuration for off-road detector.
        - traffic_light_detector : dict
            Configuration for traffic light detector.

    Attributes
    ----------
    vehicle : carla.Vehicle
        The monitored vehicle.
    print_message : bool
        Flag indicating whether to print safety messages.
    sensors : List[Any]
        List of safety sensor instances.
    """

    def __init__(self, vehicle: carla.Vehicle, params: Dict[str, Any]):
        self.vehicle = vehicle
        self.print_message: bool = params["print_message"]

        class _SafetySensor(Protocol):
            def tick(self, data_dict: Dict[str, Any]) -> None: ...

            def return_status(self) -> Dict[str, bool]: ...

            def destroy(self) -> None: ...

        self.sensors: List[_SafetySensor] = [
            CollisionSensor(vehicle, params["collision_sensor"]),
            StuckDetector(params["stuck_dector"]),
            OffRoadDetector(params["offroad_dector"]),
            TrafficLightDector(params["traffic_light_detector"], vehicle),
        ]

    def update_info(self, data_dict: Dict[str, Any]) -> None:
        """
        Update safety sensor information and return hazard status.

        Collects status from all safety sensors and optionally prints warnings
        when hazards are detected.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing current vehicle and environment data for
            sensor updates.

        Returns
        -------
        """
        status_dict: Dict[str, Any] = {}
        for sensor in self.sensors:
            sensor.tick(data_dict)
            status_dict.update(sensor.return_status())
        if self.print_message:
            print_flag = False
            # only print message when it has hazard
            for key, val in status_dict.items():
                if val:
                    print_flag = True
                    break
            if print_flag:
                logger.info(f"Safety Warning from the safety manager:\n{status_dict}")

    def destroy(self) -> None:
        for sensor in self.sensors:
            sensor.destroy()
