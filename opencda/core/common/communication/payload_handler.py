from __future__ import annotations

import logging
import pickle  # TODO: In the future pickle module will be replaced with our own safe implementation
import pathlib
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencood.communication import CommunicationDataInterface

sys.path.append(str((pathlib.Path(__file__).resolve().parent / "protos" / "cavise").resolve()))

from .protos.cavise import opencda_pb2 as proto_opencda  # noqa: E402
from .protos.cavise import artery_pb2 as proto_artery  # noqa: E402


logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.payload_handler")


class PayloadHandler:
    """Translate between CAPI protobuf messages and OpenCOOD payload state."""

    def __init__(self) -> None:
        """Initialize the transport without an attached OpenCOOD interface."""
        self.communication_interface: CommunicationDataInterface | None = None

    def bind_communication_interface(self, communication_interface: CommunicationDataInterface) -> None:
        """Bind the OpenCOOD communication state used by this transport.

        Parameters
        ----------
        communication_interface : opencood.communication.CommunicationDataInterface
            Transport-neutral state shared with OpenCOOD datasets.
        """
        self.communication_interface = communication_interface

    def make_opencda_message(self) -> proto_opencda.OpenCDAMessage:
        """Serialize outgoing OpenCOOD payloads into an OpenCDA message.

        Returns
        -------
        opencda_pb2.OpenCDAMessage
            Message containing one serialized module mapping per entity. The
            message is empty when no communication interface is bound.
        """
        opencda_message = proto_opencda.OpenCDAMessage()
        if self.communication_interface is None:
            return opencda_message

        for entity_id, payloads in self.communication_interface.get_outgoing_payloads().items():
            entity_message = opencda_message.entity.add()
            entity_message.id = entity_id
            entity_message.auxillary = pickle.dumps(payloads)

        return opencda_message

    def make_artery_payload(self, artery_message: proto_artery.ArteryMessage) -> None:
        """Insert payloads received from Artery into the OpenCOOD interface.

        Parameters
        ----------
        artery_message : artery_pb2.ArteryMessage
            CAPI message whose transmissions contain serialized module
            mappings grouped by receiver and sender.

        Raises
        ------
        TypeError
            If a decoded entity payload is not a mapping of module names to
            payloads.
        """
        if self.communication_interface is None:
            return

        for transmission in artery_message.transmissions:
            ego_id = transmission.id

            for entity_info in transmission.entity:
                if entity_info.auxillary:
                    payloads = pickle.loads(entity_info.auxillary)
                    if not isinstance(payloads, Mapping):
                        raise TypeError("Received auxiliary payload must contain a module mapping.")
                    self.communication_interface.insert_received_payloads(ego_id, entity_info.id, payloads)

    def clear_messages(self) -> None:
        """Clear payloads for the completed communication tick."""
        if self.communication_interface is not None:
            self.communication_interface.clear()
