import importlib
import pickle
import sys
import types

import pytest


class FakeEntity:
    def __init__(self) -> None:
        self.id = ""
        self.auxillary = b""


class FakeRepeatedEntity:
    def __init__(self) -> None:
        self._items: list[FakeEntity] = []

    def add(self) -> FakeEntity:
        ent = FakeEntity()
        self._items.append(ent)
        return ent

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class FakeOpenCDAMessage:
    def __init__(self) -> None:
        self.entity = FakeRepeatedEntity()


class FakeTransmission:
    def __init__(self, ego_id: str, entities: list[FakeEntity] | None = None) -> None:
        self.id = ego_id
        self.entity = list(entities or [])


class FakeArteryMessage:
    def __init__(self, transmissions: list[FakeTransmission] | None = None) -> None:
        self.transmissions = list(transmissions or [])


@pytest.fixture
def payload_handler_mod(monkeypatch):
    """
    Controlled import of payload_handler with fully stubbed protobuf modules.
    This avoids relying on protoc-generated *_pb2.py files being present on disk.
    """
    monkeypatch.setattr(sys, "path", sys.path.copy())

    fake_opencda_pb2 = types.ModuleType("opencda.core.common.communication.protos.cavise.opencda_pb2")
    fake_opencda_pb2.OpenCDAMessage = FakeOpenCDAMessage

    fake_artery_pb2 = types.ModuleType("opencda.core.common.communication.protos.cavise.artery_pb2")
    fake_artery_pb2.ArteryMessage = FakeArteryMessage

    fake_protos = types.ModuleType("opencda.core.common.communication.protos")
    fake_protos.__path__ = []

    fake_cavise = types.ModuleType("opencda.core.common.communication.protos.cavise")
    fake_cavise.__path__ = []

    setattr(fake_cavise, "opencda_pb2", fake_opencda_pb2)
    setattr(fake_cavise, "artery_pb2", fake_artery_pb2)
    setattr(fake_protos, "cavise", fake_cavise)

    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos", fake_protos)
    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos.cavise", fake_cavise)
    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos.cavise.opencda_pb2", fake_opencda_pb2)
    monkeypatch.setitem(sys.modules, "opencda.core.common.communication.protos.cavise.artery_pb2", fake_artery_pb2)

    target_mod_name = "opencda.core.common.communication.payload_handler"
    monkeypatch.delitem(sys.modules, target_mod_name, raising=False)

    return importlib.import_module(target_mod_name)


def _make_entity(entity_id: str, payload: dict) -> FakeEntity:
    ent = FakeEntity()
    ent.id = entity_id
    ent.auxillary = pickle.dumps(payload)
    return ent


class TestPayloadHandler:
    def test_set_opencda_payload_stores_module_payload(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()
        payload = {"x": 1}

        ph.set_opencda_payload("ego1", "loc", payload)

        assert ph.current_opencda_payload["ego1"]["loc"] is payload

    def test_set_opencda_payload_replaces_existing_module_payload(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()
        ph.set_opencda_payload("ego1", "loc", {"x": 1})

        ph.set_opencda_payload("ego1", "loc", {"y": 2})

        assert ph.current_opencda_payload["ego1"]["loc"] == {"y": 2}

    def test_get_artery_payload_returns_matching_module(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()
        payload = {"data": 100}
        ph.current_artery_payload = {"ego1": {"ent1": {"module": payload}}}

        assert ph.get_artery_payload("ego1", "ent1", "module") is payload

    def test_get_artery_payload_returns_none_when_missing(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()

        assert ph.get_artery_payload("ego1", "ent1", "module") is None

    def test_make_opencda_message_empty(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()

        msg = ph.make_opencda_message()
        assert hasattr(msg, "entity")
        assert len(msg.entity) == 0
        assert list(msg.entity) == []

    def test_make_opencda_message_single_entity_round_trip(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()

        ph.set_opencda_payload("ego1", "loc", {"val": 1})
        ph.set_opencda_payload("ego1", "perc", {"score": 0.5})

        msg = ph.make_opencda_message()
        assert len(msg.entity) == 1

        ent = list(msg.entity)[0]
        assert ent.id == "ego1"
        assert isinstance(ent.auxillary, (bytes, bytearray))
        assert pickle.loads(ent.auxillary) == {"loc": {"val": 1}, "perc": {"score": 0.5}}

    def test_make_opencda_message_multiple_entities_no_order_assumption(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()
        ph.current_opencda_payload = {
            "ego1": {"loc": {"val": 1}},
            "ego2": {"perc": {"val": 2}},
        }

        msg = ph.make_opencda_message()
        assert len(msg.entity) == 2

        ids = {e.id for e in msg.entity}
        assert ids == {"ego1", "ego2"}

        for e in msg.entity:
            assert isinstance(e.auxillary, (bytes, bytearray))

        decoded_by_id = {e.id: pickle.loads(e.auxillary) for e in msg.entity}
        assert decoded_by_id["ego1"] == {"loc": {"val": 1}}
        assert decoded_by_id["ego2"] == {"perc": {"val": 2}}

    def test_make_artery_payload_empty_message_does_not_change_state(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()
        ph.current_artery_payload = {"ego_existing": {"entX": {"loc": {"x": 1}}}}

        msg = FakeArteryMessage(transmissions=[])
        ph.make_artery_payload(msg)

        assert ph.current_artery_payload == {"ego_existing": {"entX": {"loc": {"x": 1}}}}

    def test_make_artery_payload_multiple_transmissions(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()

        trans1 = FakeTransmission(
            "ego1",
            [
                _make_entity("entA", {"data": 42}),
                _make_entity("entB", {"data": 43}),
            ],
        )
        trans2 = FakeTransmission(
            "ego2",
            [
                _make_entity("entC", {"info": "hello"}),
            ],
        )
        msg = FakeArteryMessage([trans1, trans2])

        ph.make_artery_payload(msg)

        assert ph.current_artery_payload["ego1"]["entA"] == {"data": 42}
        assert ph.current_artery_payload["ego1"]["entB"] == {"data": 43}
        assert ph.current_artery_payload["ego2"]["entC"] == {"info": "hello"}

    def test_make_artery_payload_accumulates_across_calls(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()

        msg1 = FakeArteryMessage([FakeTransmission("ego1", [_make_entity("entA", {"v": 1})])])
        msg2 = FakeArteryMessage([FakeTransmission("ego2", [_make_entity("entB", {"v": 2})])])

        ph.make_artery_payload(msg1)
        ph.make_artery_payload(msg2)

        assert ph.current_artery_payload["ego1"]["entA"] == {"v": 1}
        assert ph.current_artery_payload["ego2"]["entB"] == {"v": 2}

    def test_make_artery_payload_overwrite_semantics(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()

        msg1 = FakeArteryMessage([FakeTransmission("ego1", [_make_entity("entA", {"v": 1})])])
        msg2 = FakeArteryMessage([FakeTransmission("ego1", [_make_entity("entA", {"v": 2})])])

        ph.make_artery_payload(msg1)
        ph.make_artery_payload(msg2)

        assert ph.current_artery_payload["ego1"]["entA"] == {"v": 2}

    def test_clear_messages_resets_state(self, payload_handler_mod):
        ph = payload_handler_mod.PayloadHandler()
        ph.current_opencda_payload = {"a": {"m": {"x": 1}}}
        ph.current_artery_payload = {"b": {"c": {"m": {"y": 2}}}}

        ph.clear_messages()

        assert ph.current_opencda_payload == {}
        assert ph.current_artery_payload == {}
