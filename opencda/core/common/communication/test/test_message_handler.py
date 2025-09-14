import pickle
import pytest
from opencda.core.common.communication.serialize import MessageHandler
from opencda.core.common.communication.protos.cavise import capi_pb2 as proto_capi  # noqa: E402


@pytest.fixture
def mh():
    return MessageHandler()


def test_add_new_entities(mh):
    assert mh.current_message_opencda == {}
    assert mh.current_message_artery == {}

    with mh.handle_opencda_message("cav-1", "test module") as msg:
        msg["first object"] = 1
        msg["second object"] = "abc"
        msg["third object"] = [(1, "abc"), 1, "abc", []]

    assert isinstance(mh.current_message_opencda["cav-1"], dict)
    assert mh.current_message_opencda["cav-1"]["test module"]["first object"] == 1
    assert mh.current_message_opencda["cav-1"]["test module"]["second object"] == "abc"
    assert mh.current_message_opencda["cav-1"]["test module"]["third object"] == [(1, "abc"), 1, "abc", []]

    with mh.handle_artery_message("ego", "cav-2", "test module") as msg:
        msg["first object"] = 2
        msg["second object"] = "xyz"
        msg["third object"] = [(2, "xyz"), 2, "xyz", False]

    assert isinstance(mh.current_message_artery["ego"]["cav-2"], dict)
    assert mh.current_message_artery["ego"]["cav-2"]["test module"]["first object"] == 2
    assert mh.current_message_artery["ego"]["cav-2"]["test module"]["second object"] == "xyz"
    assert mh.current_message_artery["ego"]["cav-2"]["test module"]["third object"] == [(2, "xyz"), 2, "xyz", False]

    mh.clear_messages()
    assert mh.current_message_opencda == {}
    assert mh.current_message_artery == {}


def test_reuse_same_entity_module(mh):
    with mh.handle_opencda_message("cav-1", "mod") as msg:
        msg["a"] = 10

    with mh.handle_opencda_message("cav-1", "mod") as msg:
        msg["b"] = 20

    assert mh.current_message_opencda["cav-1"]["mod"]["a"] == 10
    assert mh.current_message_opencda["cav-1"]["mod"]["b"] == 20


def test_make_opencda_message_serialization_roundtrip(mh):
    with mh.handle_opencda_message("cav-123", "mod") as msg:
        msg["val"] = 42

    data = mh.make_opencda_message()
    assert isinstance(data, bytes)

    msg = proto_capi.Message()
    msg.ParseFromString(data)

    assert msg.HasField("opencda")
    assert msg.opencda.entity[0].id == "cav-123"

    aux = pickle.loads(msg.opencda.entity[0].auxillary)
    assert aux["mod"]["val"] == 42


def test_make_artery_data_with_non_artery_message_exits(mh, caplog):
    msg = proto_capi.Message(opencda=proto_capi.OpenCDAMessage())
    data = msg.SerializeToString()

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit) as e:
            mh.make_artery_data(data)

    assert e.value.code == 1
    assert "Message does not contain ArteryMessage" in caplog.text
