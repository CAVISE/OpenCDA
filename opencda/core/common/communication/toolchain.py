import os
import sys
import errno
import typing
import pathlib
import logging
import argparse
import importlib
import subprocess
import dataclasses


logger = logging.getLogger("cavise.opencda.opencda.core.common.communication.toolchain")


# Config for protoc
@dataclasses.dataclass
class MessageConfig:
    source_dir: pathlib.PurePath
    binary_dir: pathlib.PurePath


class CommunicationToolchain:
    # handle messages, it is assumed that it is safe to import messages after this call
    @staticmethod
    def handle_messages(messages: typing.List[str], config: typing.Optional[MessageConfig] = None):
        if config is None:
            config = MessageConfig(
                pathlib.PurePath("opencda/core/common/communication/messages"), pathlib.PurePath("opencda/core/common/communication/protos/cavise")
            )

        importlib.invalidate_caches()
        CommunicationToolchain.generate_message(config, messages)

    # wrap import call as boolean result, useful for running checks
    @staticmethod
    def try_import(config: MessageConfig, message: str) -> bool:
        try:
            importlib.import_module(str(config.binary_dir.joinpath(f"{message}_pb2")).replace("/", "."))
            return message in sys.modules
        except ModuleNotFoundError:
            logger.warning(f"could not found message {message}")
        return False

    # invoke subroutine to create python message impl from proto file
    @staticmethod
    def generate_message(config: MessageConfig, messages: typing.List[str]) -> None:
        command = [
            "protoc",
            f"--proto_path={config.source_dir}",
            f"--python_out={config.binary_dir}",
            *map(lambda message: config.source_dir.joinpath(f"{message}.proto"), messages),
        ]

        process = subprocess.run(
            command,
            encoding="UTF-8",
            # silent run
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )

        if process.returncode != 0:
            logger.error(
                f"failed to generate protos, subroutine exited with: {errno.errorcode[process.returncode]}\nSTDERR: {process.stderr.strip()}"
            )
            sys.exit(process.returncode)
        else:
            logger.info("generated protos for: " + " ".join(messages))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="messages", action="append")
    args = parser.parse_args()

    if args.messages is None:
        args.messages = ["capi"]

    config = MessageConfig(source_dir=pathlib.Path("messages"), binary_dir=pathlib.Path("protos/cavise"))

    print(f"using paths: {config}")

    CommunicationToolchain.handle_messages(args.messages, config)
