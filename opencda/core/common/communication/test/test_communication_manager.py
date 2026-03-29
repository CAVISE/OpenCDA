import importlib
import sys
import types
import pickle
from typing import Iterable

import pytest
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message


def _set_oneof_arm(msg: Message, field: FieldDescriptor) -> None:
    """Helper to safely set a protobuf oneof arm regardless of its schema type."""
    if field.message_type is not None:
        getattr(msg, field.name).SetInParent()
    elif field.enum_type is not None:
        setattr(msg, field.name, field.enum_type.values[0].number)
    elif field.type == FieldDescriptor.TYPE_STRING:
        setattr(msg, field.name, "x")
    elif field.type == FieldDescriptor.TYPE_BYTES:
        setattr(msg, field.name, b"x")
    elif field.type == FieldDescriptor.TYPE_BOOL:
        setattr(msg, field.name, True)
    else:
        setattr(msg, field.name, 1)


def _validate_capi_schema(capi_mod: object, opencda_mod: object) -> None:
    if not hasattr(capi_mod, "Message"):
        raise AssertionError("Generated capi_pb2 module does not expose Message.")
    if not hasattr(opencda_mod, "OpenCDAMessage"):
        raise AssertionError("Generated opencda_pb2 module does not expose OpenCDAMessage.")

    msg_cls = capi_mod.Message
    if not hasattr(msg_cls, "DESCRIPTOR"):
        raise AssertionError("proto_capi.Message must expose DESCRIPTOR.")

    desc = msg_cls.DESCRIPTOR
    if "order" not in desc.fields_by_name:
        raise AssertionError("proto_capi.Message must define an 'order' field.")

    oneofs = desc.oneofs_by_name
    if "message" not in oneofs:
        raise AssertionError("proto_capi.Message must define oneof 'message'.")

    arm_names = {f.name for f in oneofs["message"].fields}
    if "opencda" not in arm_names:
        raise AssertionError("proto_capi.Message oneof 'message' must include 'opencda' arm.")
    if "artery" not in arm_names:
        raise AssertionError("proto_capi.Message oneof 'message' must include 'artery' arm.")


def _iter_protos_modules_keys(prefix: str) -> Iterable[str]:
    for k in list(sys.modules.keys()):
        if k.startswith(prefix):
            yield k


@pytest.fixture(scope="session")
def ensure_comm_protos_ready():
    """
    Ensure protobuf Python modules required by CommunicationManager exist and match expected schema.

    If missing or stale, run CommunicationToolchain once per session to generate them. Generated
    modules may use top-level imports (e.g. `import entity_pb2`), so we temporarily add the output
    directory to sys.path for imports. If generation occurs, we delete only newly created files
    at the end of the session to avoid polluting the working tree.
    """
    import pathlib
    import shutil

    protos_dir = pathlib.Path("opencda/core/common/communication/protos/cavise")
    protos_dir.mkdir(parents=True, exist_ok=True)

    original_sys_path = list(sys.path)
    protos_path = str(protos_dir.resolve())
    if protos_path not in sys.path:
        sys.path.insert(0, protos_path)

    generated = False
    before_files = {p.name for p in protos_dir.iterdir() if p.is_file()}
    pycache = protos_dir / "__pycache__"
    before_pycache_existed = pycache.exists()
    before_pycache_files = {p.name for p in pycache.glob("*")} if before_pycache_existed else set()

    try:
        try:
            capi_mod = importlib.import_module("opencda.core.common.communication.protos.cavise.capi_pb2")
            opencda_mod = importlib.import_module("opencda.core.common.communication.protos.cavise.opencda_pb2")
            _validate_capi_schema(capi_mod, opencda_mod)
            yield
            return
        except (ImportError, AssertionError):
            # Fast-path import/validation failed; fall through to protobuf regeneration.
            pass

        if not shutil.which("protoc"):
            raise AssertionError("protoc is not installed. Please install CI deps (protoc-wheel / requirements-ci).")

        from opencda.core.common.communication.toolchain import CommunicationToolchain

        try:
            CommunicationToolchain.handle_messages(["entity", "opencda", "artery", "capi"])
        except SystemExit as e:
            raise AssertionError(f"Protobuf toolchain generation failed with exit code {e.code}") from e

        generated = True
        importlib.invalidate_caches()

        # Force Python to reload the newly generated modules instead of using stale cache.
        # 1) Clear top-level generated modules first (generated code imports these by top-level name).
        for mod_name in ["ack_pb2", "entity_pb2", "opencda_pb2", "artery_pb2", "capi_pb2"]:
            sys.modules.pop(mod_name, None)

        # 2) Clear package modules.
        for k in _iter_protos_modules_keys("opencda.core.common.communication.protos.cavise"):
            sys.modules.pop(k, None)

        capi_mod = importlib.import_module("opencda.core.common.communication.protos.cavise.capi_pb2")
        opencda_mod = importlib.import_module("opencda.core.common.communication.protos.cavise.opencda_pb2")
        _validate_capi_schema(capi_mod, opencda_mod)
        yield
    finally:
        sys.path[:] = original_sys_path

        if generated:
            after_files = {p.name for p in protos_dir.iterdir() if p.is_file()}
            created = after_files - before_files
            for name in created:
                try:
                    (protos_dir / name).unlink()
                except FileNotFoundError:
                    # File may already have been removed; ignore for best-effort cleanup.
                    pass

            # Best-effort cleanup: remove only pycache files created during this session.
            if pycache.exists():
                after_pycache_files = {p.name for p in pycache.glob("*")}
                created_pycache_files = after_pycache_files - before_pycache_files

                for name in created_pycache_files:
                    try:
                        (pycache / name).unlink()
                    except FileNotFoundError:
                        # Cache file may already have been removed; ignore for best-effort cleanup.
                        pass

                if not before_pycache_existed:
                    try:
                        pycache.rmdir()
                    except OSError:
                        # Directory may be non-empty or already gone; ignore for best-effort cleanup.
                        pass


@pytest.fixture
def fake_env(monkeypatch, ensure_comm_protos_ready):
    """
    Return an isolated environment for CommunicationManager unit tests.

    - Injects a fake `zmq` module via `sys.modules` (no real sockets / no network).
    - Uses real protobuf-generated modules (`capi_pb2`, `opencda_pb2`).
    - Ensures protobuf Python modules are available (generates them once per session via CommunicationToolchain if missing/stale).
    - Restores `sys.path` and `sys.modules` changes via pytest monkeypatch.
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
            self.getsockopt_calls = {}
            self.last_sent = None

        def setsockopt(self, opt, value):
            self.opts[opt] = value

        def getsockopt(self, opt):
            self.getsockopt_calls[opt] = self.getsockopt_calls.get(opt, 0) + 1
            if opt not in self.opts:
                raise FakeZMQError("option not set")
            return self.opts[opt]

        def connect(self, addr):
            self.connected = addr

        def send(self, data):
            self.send_calls += 1
            self.last_sent = data
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
    fake_zmq.DEALER = 1
    fake_zmq.IDENTITY = 2
    fake_zmq.SNDTIMEO = 3
    fake_zmq.RCVTIMEO = 4
    fake_zmq.ZMQError = FakeZMQError
    fake_zmq.Again = FakeAgain
    fake_zmq.Context = FakeContext

    # 3. Bind modules to sys.modules
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)

    # 4. Reload communication_manager cleanly (monkeypatch handles restoration after test)
    target_mod_name = "opencda.core.common.communication.communication_manager"
    monkeypatch.delitem(sys.modules, target_mod_name, raising=False)

    import opencda.core.common.communication.communication_manager as comm_mgr_mod

    monkeypatch.setattr(comm_mgr_mod.os, "getpid", lambda: 4242)

    return {
        "zmq": fake_zmq,
        "proto_capi": comm_mgr_mod.proto_capi,
        "proto_opencda": comm_mgr_mod.proto_opencda,
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
        assert mgr.message_order == 0
        assert mgr.socket is not None
        assert mgr.socket.connected == "tcp://127.0.0.1:5555"

        assert mgr.identity == "OpenCDA-4242"
        assert mgr.socket.opts[fake_env["zmq"].IDENTITY] == b"OpenCDA-4242"
        assert mgr.socket.opts[fake_env["zmq"].SNDTIMEO] == 1500
        assert mgr.socket.opts[fake_env["zmq"].RCVTIMEO] == 2500

    def test_init_timeout_rounding(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555", artery_send_timeout=0.001)
        assert mgr.socket.opts[fake_env["zmq"].SNDTIMEO] == 1

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

    def test_get_socket_timeout_ms_none_socket(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket = None
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
        mgr.message_order = 7  # Non-zero initial state
        msg = fake_env["proto_opencda"].OpenCDAMessage()
        ent = msg.entity.add()
        ent.id = "ego1"
        payload = {"loc": {"x": 1.0, "y": 2.0}, "meta": {"tick": 5}}
        ent.auxillary = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

        mgr.send_message(msg)
        assert mgr.socket.send_calls == 1
        assert mgr.socket.recv_calls == 0
        assert mgr.message_order == 7  # Must not increment on send
        assert mgr.socket.getsockopt_calls.get(fake_env["zmq"].SNDTIMEO, 0) == 1

        last_sent = mgr.socket.last_sent
        assert last_sent is not None

        sent = fake_env["proto_capi"].Message()
        sent.ParseFromString(last_sent)
        assert sent.order == 7
        assert sent.WhichOneof("message") == "opencda"
        assert len(sent.opencda.entity) == 1
        assert sent.opencda.entity[0].id == "ego1"
        assert isinstance(sent.opencda.entity[0].auxillary, (bytes, bytearray))
        assert pickle.loads(sent.opencda.entity[0].auxillary) == payload
        assert sent.opencda.SerializeToString() == msg.SerializeToString()

    def test_send_message_timeout_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")

        mgr.socket.send_behavior = [fake_env["zmq"].Again()]
        with pytest.raises(RuntimeError, match="Failed to send OpenCDA message to Artery"):
            mgr.send_message(fake_env["proto_opencda"].OpenCDAMessage())
        assert mgr.socket.send_calls == 1
        assert mgr.socket.getsockopt_calls.get(fake_env["zmq"].SNDTIMEO, 0) == 1

    def test_send_message_uninitialized_socket(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.socket = None

        with pytest.raises(RuntimeError, match="Socket is not initialized"):
            mgr.send_message(fake_env["proto_opencda"].OpenCDAMessage())

    def test_receive_message_success(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555", artery_receive_timeout=0.01)
        mgr.message_order = 5

        reply = fake_env["proto_capi"].Message(order=5)
        reply.artery.SetInParent()
        t = reply.artery.transmissions.add()
        t.id = "ego1"
        e = t.entity.add()
        e.id = "veh1"
        incoming_payload = {"sensor": {"speed": 3.5}, "flags": [1, 2, 3]}
        e.auxillary = pickle.dumps(incoming_payload, protocol=pickle.HIGHEST_PROTOCOL)
        mgr.socket.recv_behavior = [reply.SerializeToString()]

        result = mgr.receive_message()
        assert mgr.message_order == 6  # Increments on successful receive
        assert mgr.socket.recv_calls == 1
        assert mgr.socket.getsockopt_calls.get(fake_env["zmq"].RCVTIMEO, 0) == 1
        assert len(result.transmissions) == 1
        assert result.transmissions[0].id == "ego1"
        assert len(result.transmissions[0].entity) == 1
        assert result.transmissions[0].entity[0].id == "veh1"
        assert isinstance(result.transmissions[0].entity[0].auxillary, (bytes, bytearray))
        assert pickle.loads(result.transmissions[0].entity[0].auxillary) == incoming_payload
        assert result.DESCRIPTOR.full_name == reply.artery.DESCRIPTOR.full_name
        assert result.SerializeToString() == reply.artery.SerializeToString()

    def test_receive_message_timeout_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.message_order = 5
        mgr.socket.recv_behavior = [fake_env["zmq"].Again()]

        with pytest.raises(RuntimeError, match="receive timeout reached"):
            mgr.receive_message()
        assert mgr.message_order == 5  # Invariant
        assert mgr.socket.recv_calls == 1

    def test_receive_message_unexpected_type_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.message_order = 5

        reply = fake_env["proto_capi"].Message(order=5)
        oneof_descriptor = fake_env["proto_capi"].Message.DESCRIPTOR.oneofs_by_name["message"]
        non_artery_fields = [f for f in oneof_descriptor.fields if f.name != "artery"]
        assert non_artery_fields, "Proto schema must have at least one non-artery message arm"

        # Prefer simple scalar fields over nested messages if available
        scalars = [f for f in non_artery_fields if f.message_type is None]
        chosen_field = scalars[0] if scalars else non_artery_fields[0]
        _set_oneof_arm(reply, chosen_field)

        mgr.socket.recv_behavior = [reply.SerializeToString()]

        with pytest.raises(RuntimeError, match="unexpected message content"):
            mgr.receive_message()
        assert mgr.message_order == 5  # Invariant

    def test_receive_message_wrong_order_raises(self, fake_env):
        mgr_cls = fake_env["Manager"]
        mgr = mgr_cls("tcp://127.0.0.1:5555")
        mgr.message_order = 5

        reply = fake_env["proto_capi"].Message(order=999)
        reply.artery.SetInParent()
        mgr.socket.recv_behavior = [reply.SerializeToString()]

        with pytest.raises(RuntimeError, match="unexpected message content"):
            mgr.receive_message()
        assert mgr.message_order == 5  # Invariant

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
