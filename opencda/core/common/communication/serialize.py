"""
@serialize.py
@brief This module provides functionality for serializing and deserializing carla.Transform objects.
"""

import carla

from . import toolchain
toolchain.CommunicationToolchain.handle_messages(['opencda', 'artery'])

from .protos.cavise import opencda_pb2 as proto_opencda
from .protos.cavise import artery_pb2 as proto_artery


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


# TODO: fix docs and annotations
class MessageHandler:

    def __init__(self):
        self.opencda_message = proto_opencda.OpenCDA_message()

    def set_cav_data(self, cav_data):
        cav_message = self.opencda_message.cav.add()  # Добавляем новый объект Cav в список

        cav_message.vid = cav_data['vid']
        cav_message.ego_spd = cav_data['ego_spd']

    def serialize_to_string(self) -> str:
        message = self.opencda_message.SerializeToString()
        self.opencda_message = proto_opencda.OpenCDA_message()
        return message

    @staticmethod
    def deserialize_from_string(string):
        received_information_dict = {}

        artery_message = proto_artery.Artery_message()
        artery_message.ParseFromString(string)

        for received_info in artery_message.received_information:
            info_dict = {
                'cav_list': []
            }

            for cav_info in received_info.cav:
                cav_dict = {
                    'vid': cav_info.vid,
                    'ego_spd': cav_info.ego_spd
                }

                info_dict['cav_list'].append(cav_dict)

            received_information_dict[received_info.vid] = info_dict

        return received_information_dict
