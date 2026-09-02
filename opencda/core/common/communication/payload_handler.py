import sys
import logging
import pathlib
import pickle  # TODO: In the future pickle module will be replaced with our own safe implementation
from typing import Any

sys.path.append(str((pathlib.Path(__file__).resolve().parent / "protos" / "cavise").resolve()))

from .protos.cavise import opencda_pb2 as proto_opencda  # noqa: E402
from .protos.cavise import artery_pb2 as proto_artery  # noqa: E402


logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.payload_handler")


# TODO: fix docs and annotations
class PayloadHandler:
    def __init__(self) -> None:
        self.current_opencda_payload: dict[str, dict[str, Any]] = {}
        self.current_artery_payload: dict[str, dict[str, dict[str, Any]]] = {}

    def set_opencda_payload(self, entity_id: str, module: str, payload: object) -> None:
        """Store a module payload for the next OpenCDA message.

        Parameters
        ----------
        entity_id : str
            Identifier of the sending CAV or RSU.
        module : str
            Name of the module that owns the payload contract.
        payload : object
            Pickle-serializable module payload.
        """
        self.current_opencda_payload.setdefault(entity_id, {})[module] = payload

    def get_artery_payload(self, ego_id: str, entity_id: str, module: str) -> object | None:
        """Return a received module payload if Artery delivered it.

        Parameters
        ----------
        ego_id : str
            Identifier of the receiving ego agent.
        entity_id : str
            Identifier of the sending CAV or RSU.
        module : str
            Name of the module that owns the payload contract.

        Returns
        -------
        object | None
            Received payload, or ``None`` when no matching payload exists.
        """
        return self.current_artery_payload.get(ego_id, {}).get(entity_id, {}).get(module)

    def make_opencda_message(self) -> proto_opencda.OpenCDAMessage:
        opencda_message = proto_opencda.OpenCDAMessage()

        for entity_id in self.current_opencda_payload:
            entity_message = opencda_message.entity.add()
            entity_message.id = entity_id
            entity_message.auxillary = pickle.dumps(self.current_opencda_payload[entity_id])

        return opencda_message

    def make_artery_payload(self, artery_message: proto_artery.ArteryMessage) -> None:
        for transmission in artery_message.transmissions:
            ego_id = transmission.id
            bucket = self.current_artery_payload.setdefault(ego_id, {})

            for entity_info in transmission.entity:
                if entity_info.auxillary:
                    bucket[entity_info.id] = pickle.loads(entity_info.auxillary)

    def clear_messages(self) -> None:
        # Clear opencda and artery dict messages to avoid usage of date from previous ticks
        self.current_opencda_payload = {}
        self.current_artery_payload = {}
