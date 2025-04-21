"""
@serialize.py
@brief This module provides functionality for serializing and deserializing carla.Transform objects.
"""

import carla
from contextlib import contextmanager

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
        self.current_message_opencda = {}
        self.current_message_artery = {}

    def __serialize_ndarray(self, packed_array: dict) -> proto_opencda.NDArray:
        return proto_opencda.NDArray(
            data=packed_array['data'],
            shape=list(packed_array['shape']),
            dtype=packed_array['dtype']
        )

    def __deserialize_ndarray(self, ndarray_msg) -> dict:
        return {
            'data': ndarray_msg.data,
            'shape': list(ndarray_msg.shape),
            'dtype': ndarray_msg.dtype
        }

    def __set_entity_data(self):
        for id in self.current_message_opencda.keys():
            entity_message = self.opencda_message.entity.add()  # Добавляем новый объект entity в список
            entity_message.id = int(id)
            with self.handle_opencda_message(id, 'coperception') as msg:
                if msg['infra'] is not None:
                    entity_message.infra = msg['infra']
                if msg['velocity'] is not None:
                    entity_message.velocity = msg['velocity']
                if msg['time_delay'] is not None:
                    entity_message.time_delay = msg['time_delay']
                entity_message.object_ids.extend(msg['object_ids'])

                entity_message.object_bbx_center.CopyFrom(self.__serialize_ndarray(msg['object_bbx_center']))

                if msg['spatial_correction_matrix'] is not None:
                    entity_message.spatial_correction_matrix.CopyFrom(self.__serialize_ndarray(msg['spatial_correction_matrix']))

                if msg['voxel_num_points'] is not None:
                    entity_message.voxel_num_points.CopyFrom(self.__serialize_ndarray(msg['voxel_num_points']))
                if msg['voxel_features'] is not None:
                    entity_message.voxel_features.CopyFrom(self.__serialize_ndarray(msg['voxel_features']))
                if msg['voxel_coords'] is not None:
                    entity_message.voxel_coords.CopyFrom(self.__serialize_ndarray(msg['voxel_coords']))

                if msg['projected_lidar'] is not None:
                    entity_message.projected_lidar.CopyFrom(self.__serialize_ndarray(msg['projected_lidar']))

    @contextmanager
    def handle_opencda_message(self, id, message_type):
        if id not in self.current_message_opencda:
            self.current_message_opencda[id] = {
                'common': {
                    'ego_speed': 0
                },
                'coperception': {
                    'infra': None,
                    'velocity': None,
                    'time_delay': None,
                    'object_ids': [],
                    'object_bbx_center': {
                        'data': bytes(),
                        'shape': [],
                        'dtype': ''
                    },
                    'spatial_correction_matrix': None,
                    'voxel_num_points': None,
                    'voxel_features': None,
                    'voxel_coords': None,
                    'projected_lidar': None
                }
            }

        msg = self.current_message_opencda[id]

        if message_type in msg:
            yield msg[message_type]
        else:
            raise ValueError(
                f'Unknown message type "{message_type}". '
                'Expected one of: ["common", "coperception"]. '
                'Please verify the source of the message.'
            )

    @contextmanager
    def handle_artery_message(self, ego_id, id, message_type):
        if ego_id not in self.current_message_artery:
            self.current_message_artery[ego_id] = {}
        if id not in self.current_message_artery[ego_id]:
            self.current_message_artery[ego_id][id] = {
                'common': {
                    'ego_speed': 0
                },
                'coperception': {
                    'infra': None,
                    'velocity': None,
                    'time_delay': None,
                    'object_ids': [],
                    'object_bbx_center': {
                        'data': bytes(),
                        'shape': [],
                        'dtype': ''
                    },
                    'spatial_correction_matrix': None,
                    'voxel_num_points': None,
                    'voxel_features': None,
                    'voxel_coords': None,
                    'projected_lidar': None
                }
            }

        msg = self.current_message_artery[ego_id][id]

        if message_type in msg:
            yield msg[message_type]
        else:
            raise ValueError(
                f'Unknown message type "{message_type}". '
                'Expected one of: ["common", "coperception"]. '
                'Please verify the source of the message.'
            )

    def serialize_to_string(self) -> str:
        self.__set_entity_data()
        message = self.opencda_message.SerializeToString()
        self.opencda_message = proto_opencda.OpenCDA_message()
        return message

    def deserialize_from_string(self, string):
        artery_message = proto_artery.Artery_message()
        artery_message.ParseFromString(string)

        for received_info in artery_message.received_information:
            ego_id = received_info.id

            for entity_info in received_info.entity:
                entity_id = entity_info.id

                with self.handle_artery_message(ego_id, entity_id, 'coperception') as msg:
                    optional_fields = [
                        'infra', 'velocity', 'time_delay', 'spatial_correction_matrix',
                        'voxel_num_points', 'voxel_features', 'voxel_coords', 'projected_lidar'
                    ]

                    for field in optional_fields:
                        if entity_info.HasField(field):
                            if field in ['spatial_correction_matrix', 'voxel_num_points',
                                        'voxel_features', 'voxel_coords', 'projected_lidar']:
                                msg[field] = self.__deserialize_ndarray(getattr(entity_info, field))
                            else:
                                msg[field] = getattr(entity_info, field)
                        else:
                            msg[field] = None

                    msg['object_ids'] = list(entity_info.object_ids)
                    msg['object_bbx_center'] = self.__deserialize_ndarray(entity_info.object_bbx_center)
