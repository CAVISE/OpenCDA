import sys
import pickle  # TODO: In the future pickle module will be replaced with our own safe implementation
import logging
import pathlib
from contextlib import contextmanager

from . import toolchain

toolchain.CommunicationToolchain.handle_messages(["entity", "opencda", "artery", "capi"])

sys.path.append(str(pathlib.Path("opencda/core/common/communication/protos/cavise").resolve()))

from .protos.cavise import opencda_pb2 as proto_opencda  # noqa: E402
from .protos.cavise import capi_pb2 as proto_capi  # noqa: E402


logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.serialize")


# TODO: fix docs and annotations
class PayloadHandler:
    def __init__(self):
        self.current_opencda_payload = {}
        self.current_artery_payload = {}

    @contextmanager
    def handle_opencda_payload(self, id, module):
        if id not in self.current_opencda_payload:
            self.current_opencda_payload[id] = {module: {}}

        yield self.current_opencda_payload[id][module]

    @contextmanager
    def handle_artery_payload(self, ego_id, id, module):
        if ego_id not in self.current_artery_payload:
            self.current_artery_payload[ego_id] = {}
        if id not in self.current_artery_payload[ego_id]:
            self.current_artery_payload[ego_id][id] = {module: {}}

        yield self.current_artery_payload[ego_id][id][module]

    def make_opencda_payload(self) -> str:
        opencda_message = proto_opencda.OpenCDAMessage()

        for entity_id in self.current_opencda_payload:
            entity_message = opencda_message.entity.add()
            entity_message.id = entity_id
            entity_message.auxillary = pickle.dumps(self.current_opencda_payload[entity_id])

        msg = proto_capi.Message(opencda=opencda_message)

        return msg.SerializeToString()

    def make_artery_payload(self, string) -> None:
        msg = proto_capi.Message()
        msg.ParseFromString(string)

        if msg.WhichOneof("message") == "artery":
            artery_message = msg.artery
        else:
            logger.warning("Message does not contain ArteryMessage")
            return

        artery_message = msg.artery

        for transmission in artery_message.transmissions:
            ego_id = transmission.id

            for entity_info in transmission.entity:
                entity_id = entity_info.id
                self.current_opencda_payload[ego_id][entity_id] = pickle.loads(entity_info.auxillary)

    def clear_messages(self):
        # Clear opencda and artery dict messages to avoid usage of date from previous ticks
        self.current_opencda_payload = {}
        self.current_artery_payload = {}
