"""
@serialize.py
@brief This module provides functionality for serializing and deserializing carla.Transform objects.
"""

import carla

from . import toolchain
toolchain.CommunicationToolchain.handleMessages(['opencda', 'artery'])

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

        ego_pos_message = cav_message.ego_pos
        ego_pos_message.x = cav_data['ego_pos']['x']
        ego_pos_message.y = cav_data['ego_pos']['y']
        ego_pos_message.z = cav_data['ego_pos']['z']
        ego_pos_message.pitch = cav_data['ego_pos']['pitch']
        ego_pos_message.yaw = cav_data['ego_pos']['yaw']
        ego_pos_message.roll = cav_data['ego_pos']['roll']

        for blue_vid, blue_cav_info in cav_data['blue_vehicles'].items():
            blue_cav = cav_message.blue_vehicles.blue_cav.add()
            blue_cav.vid = blue_vid
            blue_cav.ego_spd = blue_cav_info['ego_spd']
            blue_ego_pos_message = blue_cav.ego_pos
            blue_ego_pos_message.x = blue_cav_info['ego_pos']['x']
            blue_ego_pos_message.y = blue_cav_info['ego_pos']['y']
            blue_ego_pos_message.z = blue_cav_info['ego_pos']['z']
            blue_ego_pos_message.pitch = blue_cav_info['ego_pos']['pitch']
            blue_ego_pos_message.yaw = blue_cav_info['ego_pos']['yaw']
            blue_ego_pos_message.roll = blue_cav_info['ego_pos']['roll']

        for cav_info in cav_data['vehicles']:
            cav_pos = cav_message.vehicles.cav_pos.add()
            cav_pos.x = cav_info['x']
            cav_pos.y = cav_info['y']
            cav_pos.z = cav_info['z']

        for tf_info in cav_data['traffic_lights']:
            tf_pos = cav_message.traffic_lights.tf_pos.add()
            tf_pos.x = tf_info['x']
            tf_pos.y = tf_info['y']
            tf_pos.z = tf_info['z']

        for so_info in cav_data['static_objects']:
            obj_pos = cav_message.static_objects.obj_pos.add()
            obj_pos.x = so_info['x']
            obj_pos.y = so_info['y']
            obj_pos.z = so_info['z']

        cav_message.from_who_received.extend(cav_data['from_who_received'])

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
                    'ego_spd': cav_info.ego_spd,
                    'ego_pos': {
                        'x': cav_info.ego_pos.x,
                        'y': cav_info.ego_pos.y,
                        'z': cav_info.ego_pos.z,
                        'pitch': cav_info.ego_pos.pitch,
                        'yaw': cav_info.ego_pos.yaw,
                        'roll': cav_info.ego_pos.roll
                    },
                    'blue_vehicles': [],
                    'vehicles': [],
                    'traffic_lights': [],
                    'static_objects': [],
                    'from_who_received': cav_info.from_who_received
                }

                for blue_cav_info in cav_info.blue_vehicles.blue_cav:
                    blue_cav_dict = {
                        'vid': blue_cav_info.vid,
                        'ego_spd': blue_cav_info.ego_spd,
                        'ego_pos': {
                            'x': blue_cav_info.ego_pos.x,
                            'y': blue_cav_info.ego_pos.y,
                            'z': blue_cav_info.ego_pos.z,
                            'pitch': blue_cav_info.ego_pos.pitch,
                            'yaw': blue_cav_info.ego_pos.yaw,
                            'roll': blue_cav_info.ego_pos.roll
                        }
                    }
                    cav_dict['blue_vehicles'].append(blue_cav_dict)

                for cav_pos_info in cav_info.vehicles.cav_pos:
                    cav_pos_dict = {
                        'x': cav_pos_info.x,
                        'y': cav_pos_info.y,
                        'z': cav_pos_info.z
                    }
                    cav_dict['vehicles'].append(cav_pos_dict)

                for tf_pos_info in cav_info.traffic_lights.tf_pos:
                    tf_pos_dict = {
                        'x': tf_pos_info.x,
                        'y': tf_pos_info.y,
                        'z': tf_pos_info.z
                    }
                    cav_dict['traffic_lights'].append(tf_pos_dict)

                for obj_pos_info in cav_info.static_objects.obj_pos:
                    obj_pos_dict = {
                        'x': obj_pos_info.x,
                        'y': obj_pos_info.y,
                        'z': obj_pos_info.z
                    }
                    cav_dict['static_objects'].append(obj_pos_dict)

                info_dict['cav_list'].append(cav_dict)

            received_information_dict[received_info.vid] = info_dict

        return received_information_dict
