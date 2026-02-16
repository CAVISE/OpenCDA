import zmq
import sys
import pathlib
import logging

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
        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket

        try:
            self.socket = self.context.socket(zmq.DEALER)
            logger.info("Socket has been created successfully")
            self.socket.setsockopt(zmq.SNDTIMEO, 2000)
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        except zmq.ZMQError as error:
            raise RuntimeError("Failed to create ZMQ socket") from error

        self.socket.connect(self.artery_address)
        logger.info(f"Socket is open in connect mode at {self.artery_address}")

    def send_message(self, opencda_message: proto_opencda.OpenCDAMessage) -> None:
        ack_received: bool = False
        message = proto_capi.Message(order=self.message_order, opencda=opencda_message)
        serialized_message = message.SerializeToString()

        if self.socket is None:
            raise RuntimeError("Socket is not initialized")

        self.socket.setsockopt(zmq.RCVTIMEO, 2000)

        for attempt in range(self.artery_retries):
            try:
                self.socket.send(serialized_message)
            except zmq.Again:
                logger.warning(f"Send timeout on attempt #{attempt + 1}")
                continue

            try:
                reply = self.socket.recv()
                received_message = proto_capi.Message()
                received_message.ParseFromString(reply)

                if received_message.WhichOneof("message") == "ack" and received_message.order == self.message_order:
                    logger.info("Artery received the message")
                    ack_received = True
                    break

                logger.warning("Ignoring non-ACK or wrong ACK message while waiting for true ACK")
            except zmq.Again:
                logger.warning(f"Retry #{attempt + 1}...")

        if not ack_received:
            raise RuntimeError("Failed to receive ACK from server")

    def receive_message(self) -> proto_capi.Message:
        if self.socket is None:
            raise RuntimeError("Socket is not initialized")

        self.socket.setsockopt(zmq.RCVTIMEO, max(1, int(self.artery_timeout * 1000 / self.artery_retries)))

        for attempt in range(self.artery_retries):
            try:
                reply = self.socket.recv()
            except zmq.Again:
                logger.warning(f"Result did not arrive from Artery. Retry #{attempt + 1}...")
                continue

            received_message = proto_capi.Message()
            received_message.ParseFromString(reply)

            if received_message.WhichOneof("message") != "artery" or received_message.order != self.message_order:
                logger.warning(f"Unexpected reply from Artery. Retry #{attempt + 1}...")
                continue

            self.message_order += 1
            return received_message.artery

        raise RuntimeError("Result did not arrive within timeout")

    def destroy(self) -> None:
        if self.socket:
            self.socket.close(linger=0)
            logger.info("Socket has been closed")

        self.context.term()
        logger.info("Context has been destroyed")
