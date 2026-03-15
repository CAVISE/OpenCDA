import logging
import pathlib
import typing

logger = logging.getLogger("cavise.protobuf_toolchain")


def get_proto_paths() -> typing.Dict[str, pathlib.Path]:
    base = pathlib.Path(__file__).parent
    return {
        "source": base / "messages",
        "generated": base / "protos" / "cavise",
    }


def verify_protos_built() -> bool:
    paths = get_proto_paths()
    capi_pb2 = paths["generated"] / "capi_pb2.py"

    if not capi_pb2.exists():
        logger.error(
            f"Protobuf files not found at {capi_pb2}. "
            "Please run: pip install -e . or cmake --build build"
        )
        return False

    logger.info("Protobuf files verified successfully")
    return True