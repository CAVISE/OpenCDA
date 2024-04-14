import carla
import json
from google.protobuf.json_format import MessageToDict
from opencda.core.common.OpenCDA_message_structure_pb2 import OpenCDA_message

class SerializableTransform:
    def __init__(self, transform):
        self.ego_pos = {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'roll': transform.rotation.roll
        }


    def to_dict(self):
        return self.ego_pos

    @classmethod
    def from_dict(cls, data):
        transform = carla.Transform(
            location=carla.Location(x=data['x'], y=data['y'], z=data['z']),
            rotation=carla.Rotation(pitch=data['pitch'], yaw=data['yaw'], roll=data['roll'])
        )
        return cls(transform)


def read_and_write_proto_file(proto_file, output_file): # используется для конвертации proto файла в json
    # Прочитать все блоки сообщений из файла protobuf
    with open(proto_file, "rb") as f:
        messages = f.read()

    messages = messages.split(b'CAVISE')
    messages.pop(0)

    for message_str in messages:
        message = OpenCDA_message()
        message.ParseFromString(message_str)
        message_dict = MessageToDict(message)
        with open(output_file, "a") as output_f:
            json.dump(message_dict, output_f, indent=4)
            output_f.flush()


def make_list_of_visible_cavs(proto_file, self_vid):
    with open(proto_file, "rb") as f:
        messages = f.read()

    messages = messages.split(b'CAVISE')
    messages.pop(0)
    for message_str in messages:
        message = OpenCDA_message()
        message.ParseFromString(message_str)
        if message.vid == self_vid:
            return message.from_who_received
    return []