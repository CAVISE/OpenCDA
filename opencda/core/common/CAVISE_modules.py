"""
@CAVISE_modules.py
@brief This module provides functionality for serializing and deserializing carla.Transform objects.
"""

import carla
import json
from google.protobuf.json_format import MessageToDict
from opencda.core.common.OpenCDA_message_structure_pb2 import OpenCDA_message

class SerializableTransform:
    """
    @class SerializableTransform
    @brief A class for serializing and deserializing carla.Transform objects.
    """
    def __init__(self, transform):
        """
        @brief Constructor for creating a SerializableTransform object from a carla.Transform.
        @param transform The carla.Transform object to serialize.
        """
        self.ego_pos = {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'roll': transform.rotation.roll
        }


    def to_dict(self):
        """
        @brief Serialize the transform object to a dictionary.
        @return A dictionary containing the serialized transform data.
        """
        return self.ego_pos

    @classmethod
    def from_dict(cls, data):
        """
        @brief Deserialize the transform object from a dictionary.
        @param data A dictionary containing the serialized transform data.
        @return A SerializableTransform object.
        """
        transform = carla.Transform(
            location=carla.Location(x=data['x'], y=data['y'], z=data['z']),
            rotation=carla.Rotation(pitch=data['pitch'], yaw=data['yaw'], roll=data['roll'])
        )
        return cls(transform)