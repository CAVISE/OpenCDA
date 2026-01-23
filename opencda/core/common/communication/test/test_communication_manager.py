import pytest
import zmq
from opencda.core.common.communication.manager import CommunicationManager

def _safe_close_manager(manager: CommunicationManager) -> None:
    """
    CommunicationManager API changed over time. Some versions have close_socket/close_context,
    some only expose raw `socket`/`context`. This helper closes whatever exists.
    """
    # Close socket
    if hasattr(manager, "close_socket"):
        try:
            manager.close_socket()
        except Exception:
            pass
    else:
        sock = getattr(manager, "socket", None)
        if sock is not None:
            try:
                sock.setsockopt(zmq.LINGER, 0)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    # Close context
    if hasattr(manager, "close_context"):
        try:
            manager.close_context()
        except Exception:
            pass
    else:
        ctx = getattr(manager, "context", None)
        if ctx is not None:
            try:
                ctx.term()
            except Exception:
                pass


@pytest.fixture
def zmq_pair():
    address = "tcp://127.0.0.1:5555"
    server = CommunicationManager(address)
    client = CommunicationManager(address)

    server.create_socket(zmq.PAIR, "bind")
    client.create_socket(zmq.PAIR, "connect")

    yield server, client

    _safe_close_manager(server)
    _safe_close_manager(client)


def test_socket_creation_bind_and_connect(caplog):
    address = "inproc://creation_test"
    manager = CommunicationManager(address)

    with caplog.at_level("INFO"):
        manager.create_socket(zmq.PAIR, "bind")

    assert "socket is open in bind mode" in caplog.text

    _safe_close_manager(manager)


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
    _safe_close_manager(manager)

    with caplog.at_level("ERROR"):
        manager.send_message(b"test")

    assert "error upon sending message" in caplog.text

    _safe_close_manager(manager)


def test_receive_error_logged(caplog):
    address = "inproc://bad_recv"
    manager = CommunicationManager(address)

    manager.create_socket(zmq.PAIR, "bind")
    _safe_close_manager(manager)

    with caplog.at_level("ERROR"):
        result = manager.receive_message()

    assert result is None
    assert "error upon receiving message" in caplog.text

    _safe_close_manager(manager)


def test_close_without_socket_does_not_crash():
    manager = CommunicationManager("inproc://noop")
    _safe_close_manager(manager)
