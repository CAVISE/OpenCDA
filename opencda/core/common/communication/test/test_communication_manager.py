import sys
import types

import pytest


@pytest.fixture
def fake_env(monkeypatch):
    """
    Provides a fully isolated environment with faked ZeroMQ and Protobuf modules.
    Restores sys.path and sys.modules automatically after the test.
    """
    # 1. Isolate sys.path modifications
    monkeypatch.setattr(sys, "path", sys.path.copy())

    # 2. Fake ZMQ
    class FakeZMQError(Exception):
        pass

    class FakeAgain(Exception):
        pass

    class FakeSocket:
        def __init__(self):
            self.opts = {}
            self.connected = None
            self.send_behavior = []
            self.recv_behavior = []
            self.closed = False
            self.linger = None
            self.send_calls = 0
            self.recv_calls = 0

        def setsockopt(self, opt, value):
            self.opts[opt] = value

        def getsockopt(self, opt):
            if opt not in self.opts:
                raise FakeZMQError("option not set")
            return self.opts[opt]

        def connect(self, addr):
            self.connected = addr

        def send(self, data):
            self.send_calls += 1
            if self.send_behavior:
                action = self.send_behavior.pop(0)
                if isinstance(action, Exception):
                    raise action
                return action
            return None

        def recv(self):
            self.recv_calls += 1
            if self.recv_behavior:
                action = self.recv_behavior.pop(0)
                if isinstance(action, Exception):
                    raise action
                return action
            raise Exception("Unexpected recv")

        def close(self, linger=None):
            self.closed = True
            self.linger = linger

    class FakeContext:
        def __init__(self):
            self.socket_to_return = FakeSocket()
            self.socket_error = None
            self.termed = False

        def socket(self, sock_type):
            if self.socket_error:
                raise self.socket_error
            return self.socket_to_return

        def term(self):
            self.termed = True

    fake_zmq = types.ModuleType("zmq")
    fake_zmq.DEALER = "DEALER"
    fake_zmq.IDENTITY = "IDENTITY"
    fake_zmq.SNDTIMEO = "SNDTIMEO"
    fake_zmq.RCVTIMEO = "RCVTIMEO"
    fake_zmq.ZMQError = FakeZMQError
    fake_zmq.Again = FakeAgain
    fake_zmq.Context = FakeContext

    # 3. Fake Protobuf
    class FakeMessage:
        def __init__(self, order=0, opencda=None):
            self.order = order
            self.opencda = opencda
            self._which = None
            self.artery = None

        def SerializeToString(self):
            return f"{self.order}:opencda".encode("utf-8")

        def ParseFromString(self, data):
            s = data.decode("utf-8")
            parts = s.split(":")
            self.order = int(parts[0])
            self._which = parts[1] if len(parts) > 1 else None
            if self._which == "artery":
                self.artery = "fake_artery_payload"

        def WhichOneof(self, field):
            if field == "message":
                return self._which
            return None

    class FakeOpenCDAMessage:
        pass

    fake_capi_pb2 = types.ModuleType("opencda.core.common.communication.protos.cavise.capi_pb2")
    fake_capi_pb2.Message = FakeMessage

    fake_opencda_pb2 = types.ModuleType("opencda.core.common.communication.protos.cavise.opencda_pb2")
    fake_opencda_pb2.OpenCDAMessage = FakeOpenCDAMessage

    # 4. Bind modules to sys.modules
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)

    # Create parent packages to guarantee relative imports resolve correctly
    fake_protos = types.ModuleType("opencda.core.common.communication.protos")
    fake_cavise = types.ModuleType("opencda.core.common.communication.protos.cavise")
    setattr(fake_cavise, "opencda_pb2", fake_opencda_pb2)
    setattr(fake_cavise, "capi_pb2", fake_capi_pb2)
    setattr(fake_protos, "cavise", fake_cavise)

    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos", fake_protos)
    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos.cavise", fake_cavise)
    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos.cavise.opencda_pb2", fake_opencda_pb2)
    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos.cavise.capi_pb2", fake_capi_pb2)

    # 5. Reload communication_manager cleanly (monkeypatch handles restoration after test)
    target_mod_name = "opencda.core.common.communication.communication_manager"
    monkeypatch.delitem(sys.modules, target_mod_name, raising=False)

    import opencda.core.common.communication.communication_manager as comm_mgr_mod

    monkeypatch.setattr(comm_mgr_mod.os, "getpid", lambda: 4242)

    return {
        "zmq": fake_zmq,
        "capi_pb2": fake_capi_pb2,
        "opencda_pb2": fake_opencda_pb2,
        "Manager": comm_mgr_mod.CommunicationManager,
        "Context": FakeContext,
        "Socket": FakeSocket,
    }


class TestCommunicationManager:
    def test_init_success(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555", artery_send_timeout=1.5, artery_receive_timeout=2.5)

        assert mgr.artery_address == "tcp://127.0.0.1:5555"
        assert mgr.artery_send_timeout == 1.5
        assert mgr.artery_receive_timeout == 2.5
        assert mgr.socket is not None
        assert mgr.socket.connected == "tcp://127.0.0.1:5555"

        assert mgr.identity == "OpenCDA-4242"
        assert mgr.socket.opts[fake_env["zmq"].IDENTITY] == mgr.identity.encode("utf-8")
        assert mgr.socket.opts[fake_env["zmq"].SNDTIMEO] == 1500
        assert mgr.socket.opts[fake_env["zmq"].RCVTIMEO] == 2500

    def test_init_failure(self, fake_env, monkeypatch):
        mgr_cls = fake_env["Manager"]
        zmq_mock = fake_env["zmq"]

        class ErrorContext(fake_env["Context"]):
            def socket(self, sock_type):
                raise zmq_mock.ZMQError("mock error")

        monkeypatch.setattr(zmq_mock, "Context", ErrorContext)

        with pytest.raises(RuntimeError, match="Failed to create ZMQ socket"):
            mgr_cls("tcp://127.0.0.1:5555")

    def test_get_socket_timeout_ms_reads_socket_option(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555", artery_send_timeout=1.0, artery_receive_timeout=2.0)
        assert mgr._get_socket_timeout_ms(fake_env["zmq"].SNDTIMEO, 999) == 1000
        assert mgr._get_socket_timeout_ms(fake_env["zmq"].RCVTIMEO, 999) == 2000

    def test_get_socket_timeout_ms_fallback_on_zmq_error(self, fake_env, monkeypatch):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")

        def _raise(_opt):
            raise fake_env["zmq"].ZMQError("boom")

        monkeypatch.setattr(mgr.socket, "getsockopt", _raise)
        assert mgr._get_socket_timeout_ms(fake_env["zmq"].SNDTIMEO, 123.0) == 123

    def test_format_timeout(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        assert mgr._format_timeout(-1) == "disabled"
        assert mgr._format_timeout(0) == "0 ms"
        assert mgr._format_timeout(2500) == "2500 ms"

    def test_send_message_success_sends_once(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")

        mgr.socket.send_behavior = [None]
        msg = fake_env["opencda_pb2"].OpenCDAMessage()

        mgr.send_message(msg)
        assert mgr.socket.send_calls == 1
        assert mgr.socket.recv_calls == 0
        assert mgr.message_order == 0

    def test_send_message_timeout_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")

        mgr.socket.send_behavior = [fake_env["zmq"].Again()]
        with pytest.raises(RuntimeError, match="Failed to send OpenCDA message to Artery"):
            mgr.send_message(fake_env["opencda_pb2"].OpenCDAMessage())
        assert mgr.socket.send_calls == 1

    def test_send_message_uninitialized_socket(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket = None

        with pytest.raises(RuntimeError, match="Socket is not initialized"):
            mgr.send_message(fake_env["opencda_pb2"].OpenCDAMessage())

    def test_receive_message_success(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555", artery_receive_timeout=0.01)
        mgr.socket.recv_behavior = [b"0:artery"]

        result = mgr.receive_message()
        assert result == "fake_artery_payload"
        assert mgr.message_order == 1
        assert mgr.socket.recv_calls == 1

    def test_receive_message_timeout_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket.recv_behavior = [fake_env["zmq"].Again()]

        with pytest.raises(RuntimeError, match="receive timeout reached"):
            mgr.receive_message()
        assert mgr.message_order == 0
        assert mgr.socket.recv_calls == 1

    def test_receive_message_unexpected_type_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket.recv_behavior = [b"0:ack"]

        with pytest.raises(RuntimeError, match="unexpected message content"):
            mgr.receive_message()
        assert mgr.message_order == 0

    def test_receive_message_wrong_order_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket.recv_behavior = [b"1:artery"]

        with pytest.raises(RuntimeError, match="unexpected message content"):
            mgr.receive_message()
        assert mgr.message_order == 0

    def test_receive_message_uninitialized_socket(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket = None

        with pytest.raises(RuntimeError, match="Socket is not initialized"):
            mgr.receive_message()

    def test_destroy(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")

        mgr.destroy()
        assert mgr.socket.closed is True
        assert mgr.socket.linger == 0
        assert mgr.context.termed is True

    def test_destroy_with_no_socket_still_terminates_context(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket = None

        mgr.destroy()
        assert mgr.context.termed is True
