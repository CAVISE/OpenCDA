# -*- coding: utf-8 -*-
"""
Controller interface for managing different types of vehicle controllers.
This module provides the ControlManager class which acts as a wrapper around
different controller implementations, allowing for easy switching between
different control algorithms while maintaining a consistent interface.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import importlib
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
class ControlManager(object):
    """
    Controller manager that is used to select
    and call different controller's functions.

    Parameters
    ----------
    control_config : dict
        The configuration dictionary of the control manager module.

    Attributes
    ----------
    controller : opencda object.
        The controller object of the OpenCDA framwork.
    """

    def __init__(self, control_config: Dict[str, Any]) -> None:
        """
        Initialize the control manager with the specified controller type.
        Args:
            control_config: Dictionary containing controller configuration.
        """
        controller_type = control_config["type"]
        controller = getattr(importlib.import_module("opencda.core.actuation.%s" % controller_type), "Controller")
        self.controller = controller(control_config["args"])

    def update_info(self, ego_pos: np.ndarray, ego_speed: np.ndarray) -> None:
        """
        Update ego vehicle information for controller.
        Args:
            ego_pos: Ego vehicle position.
            ego_speed: Ego vehicle speed.
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed: float, waypoint):
        """
        Execute one control step to generate vehicle control commands.
        Args:
            target_speed: The desired speed
            waypoint: The target waypoint (Transform) to navigate towards.
        Returns:
            VehicleControl: The control commands to be applied to the vehicle.
        """
        control_command = self.controller.run_step(target_speed, waypoint)
        return control_command
