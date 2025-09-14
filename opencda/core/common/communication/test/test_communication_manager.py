import pytest
import zmq
from opencda.core.common.communication.manager import CommunicationManager


@pytest.fixture
def zmq_pair():
    address = "tcp://127.0.0.1:5555"
    server = CommunicationManager(address)
    client = CommunicationManager(address)

    server.create_socket(zmq.PAIR, "bind")
    client.create_socket(zmq.PAIR, "connect")

    yield server, client

    server.close_socket()
    client.close_socket()
    server.close_context()
    client.close_context()


def test_socket_creation_bind_and_connect(caplog):
    address = "inproc://creation_test"
    manager = CommunicationManager(address)

    with caplog.at_level("INFO"):
        manager.create_socket(zmq.PAIR, "bind")

    assert "socket is open in bind mode" in caplog.text

    manager.close_socket()
    manager.close_context()


def test_send_and_receive_message(zmq_pair):
    server, client = zmq_pair

    message = b"Hello, CommunicationManager!"
    client.send_message(message)

    poller = zmq.Poller()
    poller.register(server.socket, zmq.POLLIN)
    socks = dict(poller.poll(1000))

    assert server.socket in socks, "Server did not receive message in time"

    received = server.receive_message()
    assert received == message


def test_send_error_logged(caplog):
    address = "inproc://bad_send"
    manager = CommunicationManager(address)

    manager.create_socket(zmq.PAIR, "bind")
    manager.close_socket()

    with caplog.at_level("ERROR"):
        manager.send_message(b"test")

    assert "error upon sending message" in caplog.text

    manager.close_context()


def test_receive_error_logged(caplog):
    address = "inproc://bad_recv"
    manager = CommunicationManager(address)

    manager.create_socket(zmq.PAIR, "bind")
    manager.close_socket()

    with caplog.at_level("ERROR"):
        result = manager.receive_message()

    assert result is None
    assert "error upon receiving message" in caplog.text

    manager.close_context()


def test_close_without_socket_does_not_crash():
    manager = CommunicationManager("inproc://noop")
    manager.close_socket()
    manager.close_context()
