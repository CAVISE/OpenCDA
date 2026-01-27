import zmq
import sys
import time
import pathlib
import logging
from . import toolchain

toolchain.CommunicationToolchain.handle_messages(["entity", "opencda", "artery", "capi", "ack"])

sys.path.append(str(pathlib.Path("opencda/core/common/communication/protos/cavise").resolve()))
from .protos.cavise import opencda_pb2 as proto_opencda  # noqa: E402
from .protos.cavise import capi_pb2 as proto_capi  # noqa: E402

logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.manager")


class CommunicationManager:
    def __init__(self, artery_address: str, artery_retries: int = 5, artery_timeout: int = 300) -> None:
        self.artery_address: str = artery_address
        self.artery_retries: int = artery_retries
        self.artery_timeout: int = artery_timeout
        self.message_order: int = 0
        self.context = zmq.Context()
        self.socket = None
        self.__create_socket()
        self.socket.connect(self.artery_address)
        logger.info(f"Socket is open in connect mode at {self.artery_address}")

    def __create_socket(self) -> None:
        try:
            self.socket = self.context.socket(zmq.DEALER)
            logger.info("Socket has been created successfully")
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        except zmq.ZMQError as error:
            logger.error(f"Error upon socket creation: {error}")

    def send_message(self, opencda_message: proto_opencda.OpenCDAMessage) -> None:
        ack_received = False
        message = proto_capi.Message(order=self.message_order, opencda=opencda_message)
        serialized_message = message.SerializeToString()

        for attempt in range(self.artery_retries):
            self.socket.send(serialized_message)
            try:
                reply = self.socket.recv()
                received_message = proto_capi.Message()
                received_message.ParseFromString(reply)

                if received_message.WhichOneof("message") == "ack" and received_message.order == self.message_order:
                    logger.info("Artery received the message")
                    ack_received = True
                    break
                else:
                    logger.warning("Ignoring non-ACK or wrong ACK message while waiting for true ACK")
            except zmq.Again:
                logger.warning(f"Retry #{attempt + 1}")

        if not ack_received:
            raise RuntimeError("Failed to receive ACK from server")

    def receive_message(self):
        deadline = time.monotonic() + self.artery_timeout
        artery_payload = None
        while time.monotonic() < deadline:
            try:
                reply = self.socket.recv()
                received_message = proto_capi.Message()
                received_message.ParseFromString(reply)

                if received_message.WhichOneof("message") == "artery" and received_message.order == self.message_order:
                    artery_payload = received_message.artery
                    self.message_order += 1
                    break
                else:
                    logger.warning("Unexpected reply from Artery")
            except zmq.Again:
                logger.info("Waiting for Artery message...")

        if artery_payload is None:
            raise RuntimeError("RESULT did not arrive within timeout")

        return artery_payload

    def destroy(self):
        self.socket.disconnect(self.artery_address)
        logger.info(f"Socket has been disconnected from {self.artery_address}")
        self.socket.close()
        logger.info("Socket has been closed")
        self.context.destroy()
        logger.info("Context has been destroyed")
