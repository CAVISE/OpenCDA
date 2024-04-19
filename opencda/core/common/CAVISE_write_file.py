"""
@CAVISE_write_file.py
@brief This module provides functionality for writing data to a protobuf file.
"""

from opencda.core.common.OpenCDA_message_structure_pb2 import OpenCDA_message
import yaml


class ProtobufWriter:
    """
    @class ProtobufWriter
    @brief Class for writing data to a protobuf file.
    """

    def __init__(self, filename):
        """
        @brief Constructor for ProtobufWriter class.
        @param filename The filename of the protobuf file.
        """

        self.filename = filename
        self.opencda_message = OpenCDA_message()

    def set_cav_data(self, cav_data):
        """
        @brief Sets CAV data to the protobuf message.
        @param cav_data A dictionary containing CAV data.
        """
                
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


    def write_to_file(self):
        """
        @brief Writes the protobuf message to the file.
        """

        with open("opencda/core/common/sim_param.yaml", "r") as file:
            data = yaml.safe_load(file)

        car_count = int(data["car_count"])
        counter = int(data["counter"])
        if counter >= car_count:
            self.reset_file()
            data["counter"] = 0
        else:
            data["counter"] += 1
            serialized_data = self.opencda_message.SerializeToString()
            with open(self.filename, 'ab') as f:
                f.write(serialized_data)

        with open("opencda/core/common/sim_param.yaml", "w") as file:
            yaml.dump(data, file)

    

    def reset_file(self):
        """
        @brief Resets the protobuf file and counter.
        """
        
        open(self.filename, 'w').close()
        self.record_count = 0
