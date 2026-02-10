"""
Controller interface for managing different types of vehicle controllers.

This module provides the ControlManager class which acts as a wrapper around
different controller implementations, allowing for easy switching between
different control algorithms while maintaining a consistent interface.
"""

import importlib
from typing import Any, Dict
import numpy as np
import numpy.typing as npt
import carla


class ControlManager:
    """
    Controller manager that is used to select and call different controller's functions.

    Parameters
    ----------
    control_config : Dict[str, Any]
        The configuration dictionary of the control manager module.

    Attributes
    ----------
    controller : object
        The controller object of the OpenCDA framework.
    """

    def __init__(self, control_config: Dict[str, Any]):
        controller_type = control_config["type"]
        controller = getattr(importlib.import_module("opencda.core.actuation.%s" % controller_type), "Controller")
        self.controller = controller(control_config["args"])

    def update_info(self, ego_pos: npt.NDArray[np.float32], ego_speed: npt.NDArray[np.float32]) -> None:
        """
        Update ego vehicle information for controller.

        Parameters
        ----------
        ego_pos : NDArray[np.float32]
            Ego vehicle position.
        ego_speed : NDArray[np.float32]
            Ego vehicle speed.
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed: float, waypoint: carla.Waypoint) -> carla.VehicleControl:
        """
        Execute one control step to generate vehicle control commands.

        Parameters
        ----------
        target_speed : float
            The desired speed.
        waypoint : carla.Waypoint
            The target waypoint (Transform) to navigate towards.

        Returns
        -------
        carla.VehicleControl
            The control commands to be applied to the vehicle.
        """
        control_command = self.controller.run_step(target_speed, waypoint)
        return control_command
