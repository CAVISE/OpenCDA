"""
@serialize.py
@brief This module provides functionality for serializing and deserializing carla.Transform objects.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Literal, Type, Union

from . import toolchain

toolchain.CommunicationToolchain.handle_messages(["capi"])

from .protos.cavise import capi_pb2 as proto_capi  # noqa: E402
from google.protobuf.descriptor import FieldDescriptor  # noqa: E402


# TODO: fix docs and annotations
class MessageHandler:
    """
    Handler for serializing and deserializing protobuf messages.
    
    Manages OpenCDA and Artery message serialization/deserialization,
    including support for NDArray custom types and validation of
    message field types and labels.
    
    Attributes
    ----------
    current_message_opencda : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary storing current OpenCDA messages by entity ID and module.
    current_message_artery : Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
        Dictionary storing current Artery messages by ego ID, entity ID, and module.
    TYPE_MAP : Dict[str, int]
        Mapping from type names to protobuf FieldDescriptor type constants.
    LABEL_MAP : Dict[str, int]
        Mapping from label names to protobuf FieldDescriptor label constants.
    """
    def __init__(self) -> None:
        self.current_message_opencda: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.current_message_artery: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

        self.TYPE_MAP = {
            "NDArray": FieldDescriptor.TYPE_MESSAGE,  # Custom type
            "float": FieldDescriptor.TYPE_FLOAT,
            "double": FieldDescriptor.TYPE_DOUBLE,
            "int32": FieldDescriptor.TYPE_INT32,
            "int64": FieldDescriptor.TYPE_INT64,
            "uint32": FieldDescriptor.TYPE_UINT32,
            "uint64": FieldDescriptor.TYPE_UINT64,
            "sint32": FieldDescriptor.TYPE_SINT32,
            "sint64": FieldDescriptor.TYPE_SINT64,
            "fixed32": FieldDescriptor.TYPE_FIXED32,
            "fixed64": FieldDescriptor.TYPE_FIXED64,
            "sfixed32": FieldDescriptor.TYPE_SFIXED32,
            "sfixed64": FieldDescriptor.TYPE_SFIXED64,
            "string": FieldDescriptor.TYPE_STRING,
            "bool": FieldDescriptor.TYPE_BOOL,
            "bytes": FieldDescriptor.TYPE_BYTES,
            # Add additional custom or complex types here if needed
        }

        self.LABEL_MAP = {
            "LABEL_OPTIONAL": FieldDescriptor.LABEL_OPTIONAL,
            "LABEL_REPEATED": FieldDescriptor.LABEL_REPEATED,
            "LABEL_REQUIRED": FieldDescriptor.LABEL_REQUIRED,
        }

    def __serialize_ndarray(self, packed_array: dict) -> proto_capi.NDArray:
        return proto_capi.NDArray(data=packed_array["data"], shape=list(packed_array["shape"]), dtype=packed_array["dtype"])

    def __deserialize_ndarray(self, ndarray_msg: Any) -> Dict[str, Any]:
        return {"data": ndarray_msg.data, "shape": list(ndarray_msg.shape), "dtype": ndarray_msg.dtype}

    @contextmanager
    def handle_opencda_message(self, id: str, module: str) -> Iterator[Dict[str, Any]]:
        if id not in self.current_message_opencda:
            self.current_message_opencda[id] = {module: {}}

        yield self.current_message_opencda[id][module]

    @contextmanager
    def handle_artery_message(self, ego_id: str, id: str, module: str) -> Iterator[Dict[str, Any]]:
        if ego_id not in self.current_message_artery:
            self.current_message_artery[ego_id] = {}
        if id not in self.current_message_artery[ego_id]:
            self.current_message_artery[ego_id][id] = {module: {}}

        yield self.current_message_artery[ego_id][id][module]

    def make_opencda_message(self) -> str:
        opencda_message = proto_capi.OpenCDA_message()

        for entity_id, modules_dict in self.current_message_opencda.items():
            entity_message = opencda_message.entity.add()
            entity_message.id = entity_id
            descriptor = entity_message.DESCRIPTOR

            for module_name in modules_dict.keys():
                with self.handle_opencda_message(entity_id, module_name) as msg:
                    for key, value in msg.items():
                        if key not in descriptor.fields_by_name:
                            raise ValueError(f"[{entity_id}:{module_name}] Field '{key}' not found in protobuf.")

                        if not isinstance(value, dict) or "type" not in value or "label" not in value or "data" not in value:
                            raise ValueError(f"[{entity_id}:{module_name}] Field '{key}' must have 'type', 'label', 'data'.")

                        field = descriptor.fields_by_name[key]

                        expected_type = self.TYPE_MAP.get(value["type"])
                        expected_label = self.LABEL_MAP.get(value["label"])

                        if field.type != expected_type:
                            raise ValueError(
                                f"[{entity_id}:{module_name}] Type mismatch for field '{key}': expected {field.type}, got {expected_type}"
                            )

                        if field.label != expected_label:
                            raise ValueError(
                                f"[{entity_id}:{module_name}] Label mismatch for field '{key}': expected {field.label}, got {expected_label}"
                            )

                        data = value["data"]

                        if field.type in (
                            FieldDescriptor.TYPE_INT32,
                            FieldDescriptor.TYPE_INT64,
                            FieldDescriptor.TYPE_UINT32,
                            FieldDescriptor.TYPE_UINT64,
                            FieldDescriptor.TYPE_SINT32,
                            FieldDescriptor.TYPE_SINT64,
                            FieldDescriptor.TYPE_FIXED32,
                            FieldDescriptor.TYPE_FIXED64,
                            FieldDescriptor.TYPE_SFIXED32,
                            FieldDescriptor.TYPE_SFIXED64,
                        ):
                            expected_python_type:  Union[Type[int], Type[float], Type[str], Type[bool], Literal["NDArray"]] = int
                        elif field.type in (FieldDescriptor.TYPE_FLOAT, FieldDescriptor.TYPE_DOUBLE):
                            expected_python_type = float
                        elif field.type == FieldDescriptor.TYPE_STRING:
                            expected_python_type = str
                        elif field.type == FieldDescriptor.TYPE_BOOL:
                            expected_python_type = bool
                        elif field.type == FieldDescriptor.TYPE_MESSAGE and value["type"] == "NDArray":
                            expected_python_type = "NDArray"
                        else:
                            raise ValueError(
                                f"[{entity_id}:{module_name}] Field '{key}' has unsupported protobuf field type {field.type}. "
                                "Type checking for this field is not implemented."
                            )

                        # Process NDArray type
                        if expected_python_type == "NDArray":
                            getattr(entity_message, key).CopyFrom(self.__serialize_ndarray(data))

                        # Process repeated type
                        elif expected_label == FieldDescriptor.LABEL_REPEATED:
                            if not isinstance(data, (list, tuple)):
                                raise ValueError(f"[{entity_id}:{module_name}] Field '{key}' expected list or tuple, got {type(data)}")

                            for i, item in enumerate(data):
                                if not isinstance(item, expected_python_type):
                                    raise ValueError(
                                        f"[{entity_id}:{module_name}] Field '{key}' element at index {i} expected {expected_python_type.__name__}, got {type(item).__name__}"
                                    )

                            getattr(entity_message, key).extend(data)

                        # Process scalar
                        else:
                            if not isinstance(data, expected_python_type):
                                raise ValueError(
                                    f"[{entity_id}:{module_name}] Field '{key}' expected type {expected_python_type.__name__}, got {type(data).__name__}"
                                )
                            setattr(entity_message, key, data)

        return opencda_message.SerializeToString()

    def make_artery_data(self, string: bytes) -> None:
        artery_message = proto_capi.Artery_message()
        artery_message.ParseFromString(string)

        for received_info in artery_message.received_information:
            ego_id = received_info.id

            for entity_info in received_info.entity:
                entity_id = entity_info.id
                modules_dict = self.current_message_opencda.get(entity_id, {})

                for module_name in modules_dict.keys():
                    with self.handle_artery_message(ego_id, entity_id, module_name) as msg:
                        descriptor = entity_info.DESCRIPTOR

                        for field in descriptor.fields:
                            field_name = field.name

                            has_field = True
                            if field_name != "id" and field.label != FieldDescriptor.LABEL_REPEATED:
                                has_field = entity_info.HasField(field_name)

                            if not has_field:
                                msg[field_name] = None
                                continue

                            value = getattr(entity_info, field_name)

                            if field.type == FieldDescriptor.TYPE_MESSAGE and field.message_type.name == "NDArray":
                                msg[field_name] = self.__deserialize_ndarray(value)
                            elif field.label == FieldDescriptor.LABEL_REPEATED:
                                msg[field_name] = list(value)
                            else:
                                msg[field_name] = value

    def clear_messages(self) -> None:
        # Clear opencda and artery dict messages to avoid usage of date from previous ticks
        self.current_message_opencda = {}
        self.current_message_artery = {}
