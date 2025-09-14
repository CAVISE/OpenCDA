import sys
import carla
import pickle  # TODO: In the future pickle module will be replaced with our own safe implementation
import logging
from contextlib import contextmanager

from . import toolchain

toolchain.CommunicationToolchain.handle_messages(["capi"])

from .protos.cavise import capi_pb2 as proto_capi  # noqa: E402

logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.serialize")


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
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
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
            location=carla.Location(x=data["x"], y=data["y"], z=data["z"]),
            rotation=carla.Rotation(pitch=data["pitch"], yaw=data["yaw"], roll=data["roll"]),
        )
        return cls(transform)


# TODO: fix docs and annotations
class MessageHandler:
    def __init__(self):
        self.current_message_opencda = {}
        self.current_message_artery = {}

    @contextmanager
    def handle_opencda_message(self, id, module):
        if id not in self.current_message_opencda:
            self.current_message_opencda[id] = {module: {}}

        yield self.current_message_opencda[id][module]

    @contextmanager
    def handle_artery_message(self, ego_id, id, module):
        if ego_id not in self.current_message_artery:
            self.current_message_artery[ego_id] = {}
        if id not in self.current_message_artery[ego_id]:
            self.current_message_artery[ego_id][id] = {module: {}}

        yield self.current_message_artery[ego_id][id][module]

    def make_opencda_message(self) -> str:
        opencda_message = proto_capi.OpenCDAMessage()

        for entity_id in self.current_message_opencda:
            entity_message = opencda_message.entity.add()
            entity_message.id = entity_id
            entity_message.auxillary = pickle.dumps(self.current_message_opencda[entity_id])

        msg = proto_capi.Message(opencda=opencda_message)

        return msg.SerializeToString()

    def make_artery_data(self, string) -> None:
        msg = proto_capi.Message()
        msg.ParseFromString(string)

        if msg.WhichOneof("message") == "artery":
            artery_message = msg.artery
        else:
            logger.error("Message does not contain ArteryMessage")
            sys.exit(1)

        artery_message = msg.artery

        for received_info in artery_message.received_information:
            ego_id = received_info.id

            for entity_info in received_info.entity:
                entity_id = entity_info.id
                self.current_message_opencda[ego_id][entity_id] = pickle.loads(entity_info.auxillary)

    def clear_messages(self):
        # Clear opencda and artery dict messages to avoid usage of date from previous ticks
        self.current_message_opencda = {}
        self.current_message_artery = {}
