"""
The safety manager is used to collect the AV's hazard status and give the
control back to human if necessary
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import carla

from opencda.core.safety.sensors import CollisionSensor, TrafficLightDector, StuckDetector, OffRoadDetector

logger = logging.getLogger("cavise.opencda.opencda.core.safety.safety_manager")


class SafetyManager:
    """
    A class that manages the safety of a given vehicle in a simulation environment.

    Parameters
    ----------
    vehicle: carla.Actor
        The vehicle that the SafetyManager is responsible for.
    params: dict
        A dictionary of parameters that are used to configure the SafetyManager.
    """

    def __init__(self, vehicle: carla.Vehicle, params: Mapping[str, Any], collision_sensor_actor: carla.Actor) -> None:
        self.vehicle = vehicle
        self.print_message = params["print_message"]
        self.collision_sensor = CollisionSensor.from_sensor_actor(collision_sensor_actor, params["collision_sensor"])
        self.sensors = [
            self.collision_sensor,
            StuckDetector(params["stuck_dector"]),
            OffRoadDetector(params["offroad_dector"]),
            TrafficLightDector(params["traffic_light_detector"], vehicle),
        ]

    def update_info(self, data_dict: Mapping[str, Any]) -> dict[str, Any]:
        status_dict: dict[str, Any] = {}
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
        return status_dict

    def stop_runtime_sensors(self) -> None:
        """Stop asynchronous sensor callbacks without discarding evaluation data."""
        self.collision_sensor.stop()

    def destroy(self) -> None:
        for sensor in self.sensors:
            sensor.destroy()
