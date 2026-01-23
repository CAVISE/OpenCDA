"""
Base class for sensor implementations in CARLA simulation.

This module provides the abstract base class that all sensor types inherit from,
defining the common interface for sensor data collection, visualization, and cleanup.
"""

from typing import Optional, Dict, List, Any

import cv2
import carla


class BaseSensor:
    """
    Abstract base class for all sensor types.

    Defines the common interface for sensor initialization, data visualization,
    data dumping, and cleanup operations that all sensor implementations must follow.

    Parameters
    ----------
    agent_id : str
        Unique identifier for the agent (vehicle or RSU).
    vehicle : Optional[carla.Actor]
        CARLA vehicle actor to attach sensor to. None for infrastructure sensors.
    world : carla.World
        CARLA world instance.
    config : Dict[str, Any]
        Sensor-specific configuration dictionary.
    global_position : Optional[List[float]]
        Global position [x, y, z] for infrastructure-mounted sensors.

    Attributes
    ----------
    sensor : carla.Actor
        CARLA sensor actor instance (set by subclasses).
    """

    def __init__(
        self, agent_id: str, vehicle: Optional[carla.Actor], world: carla.World, config: Dict[str, Any], global_position: Optional[List[float]]
    ):
        return

    def visualize_data(self) -> None:
        """
        Display sensor data in real-time.

        This method should be overridden by subclasses to provide sensor-specific
        visualization.
        """
        return

    def data_dump(self, output_folder: str, cur_timestamp: str) -> None:
        """
        Save sensor data to disk.

        This method should be overridden by subclasses to save sensor-specific
        data formats.

        Parameters
        ----------
        output_folder : str
            Directory path for output files.
        cur_timestamp : str
            Current timestamp for file naming.

        Returns
        -------
        None
        """
        return

    def tick(self) -> None:
        """
        Process sensor data and return metadata.

        This method should be overridden by subclasses to perform sensor-specific
        data processing and return relevant metadata.
        """
        return None

    def destroy(self) -> None:
        """
        Clean up sensor resources.

        Destroys the CARLA sensor actor and closes all OpenCV windows.

        Returns
        -------
        None
        """
        self.sensor.destroy()
        cv2.destroyAllWindows()
