import logging
import os
import pathlib
import sys

import zmq

sys.path.append(str(pathlib.Path("opencda/core/common/communication/protos/cavise").resolve()))
from .protos.cavise import capi_pb2 as proto_capi  # noqa: E402
from .protos.cavise import opencda_pb2 as proto_opencda  # noqa: E402

logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.manager")


class CommunicationManager:
    def __init__(self, artery_address: str, artery_retries: int = 5, artery_timeout: int = 300) -> None:
        self.artery_address: str = artery_address
        self.artery_retries: int = artery_retries
        self.artery_timeout: int = artery_timeout
        self.message_order: int = 0
        self.identity: str = f"OpenCDA-{os.getpid()}"
        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket

        try:
            self.socket = self.context.socket(zmq.DEALER)
            logger.info("Socket has been created successfully")
            self.socket.setsockopt(zmq.IDENTITY, self.identity.encode("utf-8"))
            self.socket.setsockopt(zmq.SNDTIMEO, 2000)
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        except zmq.ZMQError as error:
            raise RuntimeError("Failed to create ZMQ socket") from error

        self.socket.connect(self.artery_address)
        logger.info(f"Socket is open in connect mode at {self.artery_address} with identity {self.identity}")

    def send_message(self, opencda_message: proto_opencda.OpenCDAMessage) -> None:
        message = proto_capi.Message(order=self.message_order, opencda=opencda_message)
        serialized_message = message.SerializeToString()

        if self.socket is None:
            raise RuntimeError("Socket is not initialized")

        for attempt in range(self.artery_retries):
            try:
                self.socket.send(serialized_message)
                logger.info("OpenCDA message has been sent to Artery")
                return
            except zmq.Again:
                logger.warning(f"Send timeout on attempt #{attempt + 1}")

        raise RuntimeError("Failed to send OpenCDA message to Artery")

    def receive_message(self) -> proto_capi.Message:
        if self.socket is None:
            raise RuntimeError("Socket is not initialized")

        receive_timeout_ms = max(1, min(500, int(self.artery_timeout * 1000 / self.artery_retries)))
        self.socket.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)
        logger.info(f"Waiting for Artery reply with timeout {receive_timeout_ms} ms per attempt")

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
