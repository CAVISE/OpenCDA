"""
Controller interface
"""

from __future__ import annotations
import importlib

from typing import TYPE_CHECKING, Mapping, Any

if TYPE_CHECKING:
    import carla


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

    def __init__(self, control_config: Mapping[str, Any]):
        controller_type = control_config["type"]
        controller = getattr(importlib.import_module(f"opencda.core.actuation.{controller_type}"), "Controller")
        self.controller = controller(control_config["args"])

    def update_info(self, ego_pos: carla.Transform, ego_speed: float) -> None:
        """
        Update ego vehicle information for controller.
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed: float, target_location: carla.Location) -> carla.VehicleControl:
        """
        Execute current controller step.
        """
        control_command = self.controller.run_step(target_speed, target_location)
        return control_command
