import pytest
from unittest.mock import MagicMock, patch
import pathlib

from opencda.core.common.communication.toolchain import CommunicationToolchain, MessageConfig

def _norm(s) -> str:
    # Normalize paths for stable assertions (avoid platform-specific separators in stringified args)
    return str(s).replace("\\", "/")

class TestCommunicationToolchain:
    @pytest.fixture
    def mock_subprocess(self):
        # Patch specifically where it's used in the module
        with patch("opencda.core.common.communication.toolchain.subprocess.run") as mock_run:
            yield mock_run

    @pytest.fixture
    def mock_importlib(self):
        with patch("opencda.core.common.communication.toolchain.importlib.invalidate_caches") as mock_inv:
            yield mock_inv

    def test_handle_messages_default_config(self, mock_subprocess, mock_importlib):
        """Test handle_messages with default configuration."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        CommunicationToolchain.handle_messages(["msg1"])

        mock_importlib.assert_called_once()
        mock_subprocess.assert_called_once()

        args = mock_subprocess.call_args[0][0]
        # Check command structure strictly
        assert args[0] == "protoc"
        assert _norm(args[1]) == "--proto_path=opencda/core/common/communication/messages"
        assert _norm(args[2]) == "--python_out=opencda/core/common/communication/protos/cavise"
        assert "msg1.proto" in str(args[3])

    def test_handle_messages_custom_config(self, mock_subprocess, mock_importlib):
        """Test handle_messages with custom configuration."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        config = MessageConfig(source_dir=pathlib.PurePath("src"), binary_dir=pathlib.PurePath("bin"))

        CommunicationToolchain.handle_messages(["msgA", "msgB"], config)

        args = mock_subprocess.call_args[0][0]
        assert args[0] == "protoc"
        assert args[1] == "--proto_path=src"
        assert args[2] == "--python_out=bin"

        # Verify all messages are included
        joined_args = _norm(" ".join(str(a) for a in args))
        assert "src/msgA.proto" in joined_args
        assert "src/msgB.proto" in joined_args

    def test_generate_message_success_logs(self, mock_subprocess, caplog):
        """Test successful generation logs info."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        config = MessageConfig(pathlib.PurePath("s"), pathlib.PurePath("b"))

        # Check specific logger
        with caplog.at_level("INFO", logger="cavise.protobuf_toolchain"):
            CommunicationToolchain.generate_message(config, ["msg1"])

        assert "generated protos for: msg1" in caplog.text

    def test_generate_message_failure_exits(self, mock_subprocess, caplog):
        """Test failure generates error log and system exit."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "Protocol error"
        mock_subprocess.return_value = mock_process

        config = MessageConfig(pathlib.PurePath("s"), pathlib.PurePath("b"))

        # Expect sys.exit(1)
        with pytest.raises(SystemExit) as excinfo:
            with caplog.at_level("ERROR", logger="cavise.protobuf_toolchain"):
                CommunicationToolchain.generate_message(config, ["msg1"])

        assert excinfo.value.code == 1

        # Check logs
        assert "failed to generate protos" in caplog.text
        assert "STDERR: Protocol error" in caplog.text

    def test_handle_messages_empty_list(self, mock_subprocess, mock_importlib):
        """Test behavior with empty message list."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Should still run invalidate_caches and subprocess, just with fewer args
        CommunicationToolchain.handle_messages([])

        mock_importlib.assert_called_once()
        mock_subprocess.assert_called_once()

        args = mock_subprocess.call_args[0][0]
        # args should be: ["protoc", "--proto_path=...", "--python_out=..."]
        # with no further arguments for files
        assert len(args) == 3
        assert args[0] == "protoc"
