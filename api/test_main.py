"""API lifecycle regression tests; no simulator installation is required."""

import asyncio
import importlib.util
import logging
from collections.abc import Coroutine, Iterator
from types import ModuleType
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> ModuleType:
    spec = importlib.util.spec_from_file_location("component_api", Path(__file__).with_name("main.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    caplog.set_level(logging.DEBUG, logger=module.logger.name)
    if hasattr(module, "_group_exists"):
        monkeypatch.setattr(module, "_group_exists", lambda pid: False)
    return module


@pytest.fixture
def client(api: ModuleType) -> Iterator[TestClient]:
    with TestClient(api.app) as client:
        yield client


@pytest.fixture
def process(api: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Mock:
    process = Mock(pid=12345, returncode=None)

    async def wait() -> int:
        process.returncode = 0
        return 0

    process.wait = AsyncMock(side_effect=wait)
    monkeypatch.setattr(api, "process", process)
    if hasattr(api, "os"):
        monkeypatch.setattr(api.os, "killpg", process.send_signal)
    return process


def test_idle(client: TestClient) -> None:
    assert client.get("/v1/status").json()["state"] == "IDLE"
    assert client.post("/v1/stop").json()["state"] == "IDLE"


def test_stop_and_repeat(api: ModuleType, client: TestClient, process: Mock, caplog: pytest.LogCaptureFixture) -> None:
    assert client.post("/v1/stop").json() == {"state": "STOPPED", "pid": 12345, "returncode": 0}
    assert api.process is None
    assert client.post("/v1/stop").status_code == 200
    assert process.send_signal.call_count == 1
    assert "Process stopped" in caplog.text


def test_disappeared_process(api: ModuleType, client: TestClient, process: Mock, caplog: pytest.LogCaptureFixture) -> None:
    process.send_signal.side_effect = ProcessLookupError()
    assert client.post("/v1/stop").status_code == 200
    assert api.process is None
    assert "already exited" in caplog.text


def test_signal_failure_can_retry(api: ModuleType, client: TestClient, process: Mock, caplog: pytest.LogCaptureFixture) -> None:
    process.send_signal.side_effect = PermissionError("private detail")
    response = client.post("/v1/stop")
    assert response.status_code == 500
    assert "private detail" not in response.text
    assert api.process is process and not api.stop_requested
    assert any(record.exc_info for record in caplog.records)
    process.send_signal.side_effect = None
    assert client.post("/v1/stop").status_code == 200


def test_timeout_can_retry(
    api: ModuleType, client: TestClient, process: Mock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def timeout(awaitable: Coroutine[Any, Any, Any], timeout: float) -> None:
        awaitable.close()
        raise asyncio.TimeoutError()

    with monkeypatch.context() as patch:
        patch.setattr(api.asyncio, "wait_for", timeout)
        assert client.post("/v1/stop").status_code == 504
    assert api.process is process and api.stop_requested
    assert "Stop timed out" in caplog.text
    assert client.post("/v1/stop").status_code == 200
    assert process.send_signal.call_count == 1


def test_start_and_duplicate(api: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    spawn = AsyncMock(return_value=Mock(pid=12345, returncode=None))
    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    assert client.post("/v1/run?scenario=demo").status_code in (200, 202)
    assert client.get("/v1/status").json()["pid"] == 12345
    assert client.post("/v1/run?scenario=demo").status_code in (200, 409)
    assert spawn.await_count == 1
    assert "started" in caplog.text


def test_start_failure_can_retry(api: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    spawn = AsyncMock(side_effect=FileNotFoundError("private executable"))
    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    response = client.post("/v1/run?scenario=demo")
    assert response.status_code == 500
    assert "private executable" not in response.text
    assert api.process is None
    assert any(record.exc_info for record in caplog.records)
    spawn.side_effect = None
    spawn.return_value = Mock(pid=12345, returncode=None)
    assert client.post("/v1/run?scenario=demo").status_code in (200, 202)


@pytest.mark.parametrize(
    "name,content,code",
    [
        ("demo.yaml", b"[", 422),
        ("demo.yaml", b"[]", 422),
        ("demo.txt", b"a: 1", 422),
        ("demo.yaml", b"x" * (1024 * 1024 + 1), 413),
    ],
)
def test_upload_validation(client: TestClient, name: str, content: bytes, code: int, caplog: pytest.LogCaptureFixture) -> None:
    assert client.post("/v1/run", files={"config": (name, content)}).status_code == code
    assert "rejected" in caplog.text


@pytest.mark.parametrize("query", ["", "?scenario=", "?scenario=%20"])
def test_name_validation(client: TestClient, query: str) -> None:
    assert client.post("/v1/run" + query).status_code == 422


def test_upload_save_failure_and_retry(
    api: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(api, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", AsyncMock(return_value=Mock(pid=12345, returncode=None)))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", Mock(side_effect=OSError("disk full")))
        assert client.post("/v1/run", files={"config": ("demo.yaml", b"a: 1")}).status_code == 500
    assert api.process is None
    assert any(record.exc_info for record in caplog.records)
    assert client.post("/v1/run", files={"config": ("demo.yaml", b"a: 1")}).status_code == 202
