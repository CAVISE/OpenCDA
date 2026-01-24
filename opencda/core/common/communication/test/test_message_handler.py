import pytest
from opencda.core.common.communication.serialize import MessageHandler
from opencda.core.common.communication.protos.cavise import capi_pb2 as proto_capi  # noqa: E402
from google.protobuf.descriptor import FieldDescriptor  # noqa: E402


@pytest.fixture
def mh():
    return MessageHandler()


def _pick_entity_field_and_payload():
    """
    Pick a real protobuf field (other than 'id') from OpenCDA_message.entity
    and generate a valid MessageHandler payload for it.

    This keeps the test stable even if the proto schema changes.
    """
    ent = proto_capi.OpenCDA_message().entity.add()
    desc = ent.DESCRIPTOR

    type_map = {
        FieldDescriptor.TYPE_INT32: ("int32", 42),
        FieldDescriptor.TYPE_INT64: ("int64", 42),
        FieldDescriptor.TYPE_UINT32: ("uint32", 42),
        FieldDescriptor.TYPE_UINT64: ("uint64", 42),
        FieldDescriptor.TYPE_SINT32: ("sint32", 42),
        FieldDescriptor.TYPE_SINT64: ("sint64", 42),
        FieldDescriptor.TYPE_FLOAT: ("float", 1.25),
        FieldDescriptor.TYPE_DOUBLE: ("double", 1.25),
        FieldDescriptor.TYPE_STRING: ("string", "abc"),
        FieldDescriptor.TYPE_BOOL: ("bool", True),
    }

    for f in desc.fields:
        if f.name == "id":
            continue

        # NDArray custom message support (if present in schema)
        if f.type == FieldDescriptor.TYPE_MESSAGE and getattr(f.message_type, "name", None) == "NDArray":
            payload = {
                "type": "NDArray",
                "label": "LABEL_OPTIONAL" if f.label != FieldDescriptor.LABEL_REPEATED else "LABEL_REPEATED",
                "data": {"data": b"\x00\x01", "shape": (1, 2), "dtype": "uint8"},
            }
            return f, payload

        if f.type not in type_map:
            continue

        type_str, scalar = type_map[f.type]
        if f.label == FieldDescriptor.LABEL_REPEATED:
            payload = {"type": type_str, "label": "LABEL_REPEATED", "data": [scalar]}
        else:
            # proto3 typically uses optional semantics; MessageHandler expects LABEL_OPTIONAL
            payload = {"type": type_str, "label": "LABEL_OPTIONAL", "data": scalar}

        return f, payload

    pytest.skip("No suitable protobuf fields found for OpenCDA_message.entity (other than id).")


def test_add_new_entities(mh):
    assert mh.current_message_opencda == {}
    assert mh.current_message_artery == {}

    with mh.handle_opencda_message("cav-1", "test module") as msg:
        # This test only checks that the nested dict structure exists and can store values.
        msg["dummy"] = {"type": "int32", "label": "LABEL_OPTIONAL", "data": 1}

    assert isinstance(mh.current_message_opencda["cav-1"], dict)
    assert mh.current_message_opencda["cav-1"]["test module"]["dummy"]["data"] == 1

    with mh.handle_artery_message("ego", "cav-2", "test module") as msg:
        msg["dummy"] = 2

    assert isinstance(mh.current_message_artery["ego"]["cav-2"], dict)
    assert mh.current_message_artery["ego"]["cav-2"]["test module"]["dummy"] == 2

    mh.clear_messages()
    assert mh.current_message_opencda == {}
    assert mh.current_message_artery == {}


def test_reuse_same_entity_module(mh):
    with mh.handle_opencda_message("cav-1", "mod") as msg:
        msg["a"] = {"type": "int32", "label": "LABEL_OPTIONAL", "data": 10}

    with mh.handle_opencda_message("cav-1", "mod") as msg:
        msg["b"] = {"type": "int32", "label": "LABEL_OPTIONAL", "data": 20}

    assert mh.current_message_opencda["cav-1"]["mod"]["a"]["data"] == 10
    assert mh.current_message_opencda["cav-1"]["mod"]["b"]["data"] == 20


def test_make_opencda_message_serialization_roundtrip(mh):
    field, payload = _pick_entity_field_and_payload()

    with mh.handle_opencda_message("cav-123", "mod") as msg:
        msg[field.name] = payload

    data = mh.make_opencda_message()
    assert isinstance(data, bytes)

    parsed = proto_capi.OpenCDA_message()
    parsed.ParseFromString(data)

    assert len(parsed.entity) >= 1
    assert parsed.entity[0].id == "cav-123"

    # Validate the field made it into protobuf
    ent0 = parsed.entity[0]
    if field.label == FieldDescriptor.LABEL_REPEATED:
        assert list(getattr(ent0, field.name)) == payload["data"]
    elif field.type == FieldDescriptor.TYPE_MESSAGE and getattr(field.message_type, "name", None) == "NDArray":
        nd = getattr(ent0, field.name)
        assert bytes(nd.data) == payload["data"]["data"]
        assert list(nd.shape) == list(payload["data"]["shape"])
        assert nd.dtype == payload["data"]["dtype"]
    else:
        assert getattr(ent0, field.name) == payload["data"]


def test_make_artery_data_populates_current_message_artery(mh):
    # make_artery_data only fills modules that exist in current_message_opencda for a given entity_id
    with mh.handle_opencda_message("cav-123", "mod") as _:
        pass

    artery = proto_capi.Artery_message()
    ri = artery.received_information.add()
    ri.id = "ego"
    ent = ri.entity.add()
    ent.id = "cav-123"

    mh.make_artery_data(artery.SerializeToString())

    assert "ego" in mh.current_message_artery
    assert "cav-123" in mh.current_message_artery["ego"]
    assert "mod" in mh.current_message_artery["ego"]["cav-123"]
    assert mh.current_message_artery["ego"]["cav-123"]["mod"]["id"] == "cav-123"
