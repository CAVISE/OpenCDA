"""
@CAVISE_read_file.py
@brief This module provides functionality for extracting information from a protobuf file.
"""

import os
from opencda.core.common.Artery_message_structure_pb2 import Artery_message


def extract_info_from_proto_file(proto_file):
    """
    @brief Extracts information from a protobuf file.
    @param proto_file Path to the protobuf file.
    @return A dictionary containing the extracted information.
    """
    received_information_dict = {}

    artery_message = Artery_message()

    with open(proto_file, 'rb') as f:
        artery_message.ParseFromString(f.read())

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