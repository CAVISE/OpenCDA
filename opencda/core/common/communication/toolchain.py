import os
import sys
import errno
import typing
import pathlib
import logging
import importlib
import subprocess
import dataclasses

logger = logging.getLogger('cavise.protobuf_toolchain')

# Config for protoc
@dataclasses.dataclass
class MessageConfig:
    source_dir: pathlib.PurePath
    binary_dir: pathlib.PurePath

class CommunicationToolchain:

    # handle messages, it is assumed that it is safe to import messages after this call
    @staticmethod
    def handleMessages(messages: typing.List[str], config: MessageConfig | None = None):
        if config is None:
            config = MessageConfig(
                pathlib.PurePath('opencda/core/common/communication/messages'),
                pathlib.PurePath('opencda/core/common/communication/protos/cavise')
            )

        importlib.invalidate_caches()
        for message in messages:
            if not CommunicationToolchain.tryImport(config, message):
                CommunicationToolchain.generateMessage(config, [message])
            else:
                logger.info(f'found generated message: {message}')

    
    # wrap import call as boolean result, useful for running checks
    @staticmethod
    def tryImport(config: MessageConfig, message: str) -> bool:
        try:
            importlib.import_module(str(config.binary_dir.joinpath(f"{message}_pb2")).replace('/', '.'))
            return message in sys.modules
        except ModuleNotFoundError:
            logger.warning(f'could not found message {message}')
        return False
            

    # invoke subroutine to create python message impl from proto file
    @staticmethod
    def generateMessage(config: MessageConfig, messages: typing.List[str]) -> None:
        process = subprocess.run(
            [
                'protoc', 
                f'--proto_path={config.source_dir}', 
                f'--python_out={config.binary_dir}', 
                *map(lambda message: config.source_dir.joinpath(f'{message}.proto'), messages)
            ], 
            encoding='UTF-8',
            # silent run
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, 
            env=os.environ
        )

        if process.returncode != 0:
            logger.error(f'failed to generate protos, subroutine exited with: {errno.errorcode[process.returncode]}')
            sys.exit(process.returncode)
        else:
            logger.info('generated protos for: ' + ' '.join(messages))