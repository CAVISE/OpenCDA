import logging
import os
import pathlib
import sys

import zmq

sys.path.append(str(pathlib.Path("opencda/core/common/communication/protos/cavise").resolve()))
from .protos.cavise import capi_pb2 as proto_capi  # noqa: E402
from .protos.cavise import opencda_pb2 as proto_opencda  # noqa: E402

logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.communication_manager")


class CommunicationManager:
    def __init__(self, artery_address: str, artery_send_timeout: float = 5.0, artery_receive_timeout: float = 300.0) -> None:
        self.artery_address: str = artery_address
        self.message_order: int = 0
        self.artery_send_timeout = artery_send_timeout
        self.artery_receive_timeout = artery_receive_timeout
        self.identity: str = f"OpenCDA-{os.getpid()}"
        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket

        try:
            self.socket = self.context.socket(zmq.DEALER)
            logger.info("Created ZMQ DEALER socket for Artery communication")
            self.socket.setsockopt(zmq.IDENTITY, self.identity.encode("utf-8"))
            self.socket.setsockopt(zmq.SNDTIMEO, int(self.artery_send_timeout * 1000))
            self.socket.setsockopt(zmq.RCVTIMEO, int(self.artery_receive_timeout * 1000))
        except zmq.ZMQError as error:
            raise RuntimeError("Failed to create ZMQ socket") from error

        self.socket.connect(self.artery_address)
        logger.info("Connected to Artery at %s (identity=%s)", self.artery_address, self.identity)

    def _get_socket_timeout_ms(self, option: int, fallback_timeout: float) -> int:
        if self.socket is None:
            return int(fallback_timeout)

        try:
            return int(self.socket.getsockopt(option))
        except zmq.ZMQError:
            return int(fallback_timeout)

    @staticmethod
    def _format_timeout(timeout_ms: int) -> str:
        return "disabled" if timeout_ms < 0 else f"{timeout_ms} ms"

    def send_message(self, opencda_message: proto_opencda.OpenCDAMessage) -> None:
        message = proto_capi.Message(order=self.message_order, opencda=opencda_message)
        serialized_message = message.SerializeToString()
        send_timeout_ms = self._get_socket_timeout_ms(zmq.SNDTIMEO, self.artery_send_timeout)

        if self.socket is None:
            raise RuntimeError("Socket is not initialized")

        try:
            self.socket.send(serialized_message)
            logger.info("Sent OpenCDA message to Artery (order=%s)", self.message_order)
            return
        except zmq.Again:
            logger.warning(
                "Failed to send OpenCDA message to Artery: socket send timeout reached (%s)",
                self._format_timeout(send_timeout_ms),
            )

        raise RuntimeError("Failed to send OpenCDA message to Artery")

    def receive_message(self) -> proto_capi.Message:
        if self.socket is None:
            raise RuntimeError("Socket is not initialized")

        receive_timeout_ms = self._get_socket_timeout_ms(zmq.RCVTIMEO, self.artery_receive_timeout)
        logger.info(
            "Waiting for reply from Artery (order=%s, socket receive timeout=%s)",
            self.message_order,
            self._format_timeout(receive_timeout_ms),
        )
        try:
            reply = self.socket.recv()
        except zmq.Again:
            logger.warning(
                "No reply from Artery before socket receive timeout (%s)",
                self._format_timeout(receive_timeout_ms),
            )
            raise RuntimeError("Failed to receive reply from Artery: receive timeout reached")

        received_message = proto_capi.Message()
        received_message.ParseFromString(reply)

        if received_message.WhichOneof("message") != "artery" or received_message.order != self.message_order:
            logger.warning(
                "Unexpected reply from Artery (expected message='artery', order=%s; got message='%s', order=%s)",
                self.message_order,
                received_message.WhichOneof("message"),
                received_message.order,
            )
            raise RuntimeError("Failed to process reply from Artery: unexpected message content")

        self.message_order += 1
        logger.info("Received valid reply from Artery (order=%s)", received_message.order)
        return received_message.artery

    def destroy(self) -> None:
        if self.socket:
            self.socket.close(linger=0)
            logger.info("Closed Artery socket")

        self.context.term()
        logger.info("Destroyed ZMQ context")
