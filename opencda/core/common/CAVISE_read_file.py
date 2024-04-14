import os
from opencda.core.common.Artery_message_structure_pb2 import Artery_message


def extract_info_from_proto_file(proto_file):
    received_information_dict = {}

    # Создаем объект сообщения из protobuf файла
    artery_message = Artery_message()

    # Прочитываем содержимое protobuf файла и разбираем его
    with open(proto_file, 'rb') as f:
        artery_message.ParseFromString(f.read())

    # Извлекаем информацию о каждом сообщении Received_information
    for received_info in artery_message.received_information:
        info_dict = {
            'cav_list': []
        }

        # Извлекаем информацию о каждом Cav в сообщении
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

            # Извлекаем информацию о каждой BlueCav в Cav
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

            # Извлекаем информацию о каждой CavPos в Vehicles
            for cav_pos_info in cav_info.vehicles.cav_pos:
                cav_pos_dict = {
                    'x': cav_pos_info.x,
                    'y': cav_pos_info.y,
                    'z': cav_pos_info.z
                }
                cav_dict['vehicles'].append(cav_pos_dict)

            # Извлекаем информацию о каждой TfPos в TrafficLights
            for tf_pos_info in cav_info.traffic_lights.tf_pos:
                tf_pos_dict = {
                    'x': tf_pos_info.x,
                    'y': tf_pos_info.y,
                    'z': tf_pos_info.z
                }
                cav_dict['traffic_lights'].append(tf_pos_dict)

            # Извлекаем информацию о каждой ObjPos в StaticObjects
            for obj_pos_info in cav_info.static_objects.obj_pos:
                obj_pos_dict = {
                    'x': obj_pos_info.x,
                    'y': obj_pos_info.y,
                    'z': obj_pos_info.z
                }
                cav_dict['static_objects'].append(obj_pos_dict)

            # Добавляем Cav в список Cav данного сообщения
            info_dict['cav_list'].append(cav_dict)

        # Добавляем информацию об этом сообщении в общий словарь
        received_information_dict[received_info.vid] = info_dict

    return received_information_dict


if __name__ == "__main__":
    abs_proto_file_path = os.path.abspath('core/common/Messages/Artery_message.proto')
    info_list = extract_info_from_proto_file(abs_proto_file_path)
    print(info_list['41554974829481398603018559275587605542']['cav_list'])
