import zmq
import logging

logger = logging.getLogger("cavise.communication")


class CommunicationManager:
    """
    Manager for ZeroMQ socket communication.

    Handles creation, binding/connecting, and message exchange over ZeroMQ sockets.
    Supports both server (bind) and client (connect) modes.

    Parameters
    ----------
    address : str

    Attributes
    ----------
    address : str
        ZeroMQ socket address.
    context : zmq.Context
        ZeroMQ context for socket creation.
    socket : Optional[zmq.Socket]
        ZeroMQ socket instance, None until created.
    """

    def __init__(self, address: str) -> None:
        self.address = address
        self.context = zmq.Context()
        self.socket = None

    def create_socket(self, socket_type: int, start_func: str) -> None:
        """
        Create and initialize ZeroMQ socket.

        Creates socket of specified type and either binds (server mode)
        or connects (client mode) to the configured address.

        Parameters
        ----------
        socket_type : int
            ZeroMQ socket type constant (e.g., zmq.REQ, zmq.REP, zmq.PUB, zmq.SUB).
        start_func : str
            Socket initialization mode: "bind" for server, "connect" for client.

        Raises
        ------
        zmq.ZMQError
            If socket creation, binding, or connection fails.
        """
        try:
            self.socket = self.context.socket(socket_type)
            if start_func == "bind":
                self.socket.bind(self.address)  # NOTE None-check is required
                logger.info(f"socket is open in {start_func} mode at {self.address}")
            if start_func == "connect":
                self.socket.connect(self.address)  # NOTE None-check is required
                logger.info(f"socket is open in {start_func} mode at {self.address}")
        except zmq.ZMQError as error:
            logger.error(f"error upon socket creation: {error}")

    def send_message(self, message: str | bytes) -> None:
        """
        Send message through ZeroMQ socket.

        Parameters
        ----------
        message : str
            Message string to send.

        Raises
        ------
        zmq.ZMQError
            If message sending fails.
        AssertionError
            If socket is not initialized.
        """
        try:
            self.socket.send(message)  # NOTE None-check is required
            logger.info(f"message sent to {self.address}")
        except zmq.ZMQError as error:
            logger.error(f"error upon sending message: {error}")

    def receive_message(self) -> bytes:  # NOTE Need to return empty bytes
        """
        Receive message from ZeroMQ socket.

        Returns
        -------
        bytes
            Received message data, or empty bytes if error occurs.

        Raises
        ------
        zmq.ZMQError
            If message receiving fails.
        AssertionError
            If socket is not initialized.
        """
        try:
            received_data = self.socket.recv()
            logger.info(f"message received from {self.address}")
            return received_data
        except zmq.ZMQError as error:
            logger.error(f"error upon receiving message: {error}")
