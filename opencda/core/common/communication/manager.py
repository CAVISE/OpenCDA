import zmq
import logging

logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.manager")


class CommunicationManager:
    def __init__(self, address: str) -> None:
        self.address = address
        self.context = zmq.Context()
        self.socket = None

    def create_socket(self, socket_type: int, start_func: str) -> None:
        try:
            self.socket = self.context.socket(socket_type)
            if start_func == "bind":
                self.socket.bind(self.address)
                logger.info(f"socket is open in {start_func} mode at {self.address}")
            if start_func == "connect":
                self.socket.connect(self.address)
                logger.info(f"socket is open in {start_func} mode at {self.address}")
        except zmq.ZMQError as error:
            logger.error(f"error upon socket creation: {error}")
