import os
import zmq
import logging

logger = logging.getLogger('cavise.communication')


class CommunicationManager:
    def __init__(self, address: str) -> None:
        self.address = address
        self.context = zmq.Context()
        self.socket = None

    def create_socket(self, socket_type: int, start_func: str) -> None:
        try:
            self.socket = self.context.socket(socket_type)
            if start_func == 'bind':
                self.socket.bind(self.address)
                logger.info(f'socket is open in {start_func} mode at {self.address}')
            if start_func == 'connect':
                self.socket.connect(self.address)
                logger.info(f'socket is open in {start_func} mode at {self.address}')
        except zmq.ZMQError as error:
            logger.error(f'error upon socket creation: {error}')

    def send_message(self, message: str) -> None:
        try:
            self.socket.send(message)
            logger.info(f'message sent to {self.address}')
        except zmq.ZMQError as error:
            logger.error(f'error upon sending message: {error}')

    def receive_message(self) -> bytes:
        try:
            received_data = self.socket.recv()
            logger.info(f'message received from {self.address}')
            return received_data
        except zmq.ZMQError as error:
            logger.error(f'error upon receiving message: {error}')

    # TODO: maybe add with (python context manager) support? 
    def close_socket(self) -> None:
        if self.socket:
            self.socket.close()

    def close_context(self) -> None:
        if self.context:
            self.context.term()
