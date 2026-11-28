"""Runner contract tests use private storage and fake execution, never simulators."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api import main
from api.runner import Runner
from api.state import Store, process_identity


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Serve a configured binding while replacing only outbound enrollment."""
    monkeypatch.setenv("CAVISE_RUNNER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("CAVISE_REGISTRATION_ID", str(uuid4()))
    monkeypatch.setenv("CAVISE_REGISTRATION_TOKEN", "test-secret")
    monkeypatch.setenv("CAVISE_MANAGER_URL", "http://manager")
    monkeypatch.setenv("CAVISE_RUNNER_API_URL", "http://runner")
    monkeypatch.setattr(Runner, "enroll", AsyncMock())
    with TestClient(main.app, headers={"Authorization": "Bearer test-secret"}) as client:
        yield client


def test_snapshot_and_authentication(client):
    """The snapshot matches Manager fields; credentials never authorize public reads implicitly."""
    snapshot = client.get("/v1/status").json()
    assert snapshot["occupancy"] == "idle" and snapshot["current_task"] is None
    assert snapshot["registration_id"] != snapshot["runner_id"] != snapshot["instance_id"]
    assert client.get("/v1/status").json() == snapshot
    assert "test-secret" not in str(snapshot)
    for method, path in [
        ("GET", "/v1/status"),
        ("GET", "/v1/tasks/a"),
        ("POST", "/v1/run"),
        ("POST", "/v1/stop"),
        ("DELETE", "/v1/registration"),
    ]:
        response = client.request(method, path, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
        assert set(response.json()) == {"code", "message"}
    assert client.get("/healthz", headers={"Authorization": ""}).json() == {"status": "ok"}


def test_revocation_and_retired_token(client):
    """A lost acknowledgement can be retried, while every ordinary route rejects the retired token."""
    response = client.delete("/v1/registration")
    assert response.status_code == 204
    response = client.delete("/v1/registration")
    assert response.status_code == 204
    assert client.get("/v1/status").status_code == 401
    response = client.post("/v1/run?scenario=demo")
    assert response.status_code == 401


def test_start_duplicate_and_busy_unbinding(client, monkeypatch):
    """Accepted preparation is durable and prevents a second run or unbinding."""
    spawn = AsyncMock(return_value=Mock(pid=os.getpid()))
    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", spawn)
    response = client.post("/v1/run?scenario=demo")
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert client.get("/v1/status").json()["current_task"] == {"id": task_id, "state": "PREPARING"}
    response = client.post("/v1/run?scenario=demo")
    assert response.status_code == 409
    response = client.delete("/v1/registration")
    assert response.status_code == 409
    assert spawn.await_count == 1


def test_spawn_failure_releases_reservation(client, monkeypatch):
    """A failed exec records failure and leaves the node available for correction."""
    spawn = AsyncMock(side_effect=FileNotFoundError("private executable"))
    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", spawn)
    response = client.post("/v1/run?scenario=demo")
    assert response.status_code == 500 and "private executable" not in response.text
    assert client.get("/v1/status").json()["occupancy"] == "idle"


@pytest.mark.parametrize(
    "name,content,code",
    [
        ("demo.yaml", b"[", 422),
        ("demo.yaml", b"[]", 422),
        ("demo.txt", b"a: 1", 422),
        ("demo.yaml", b"x" * (1024 * 1024 + 1), 413),
    ],
)
def test_upload_validation(client, name, content, code):
    """Invalid uploads do not reserve execution or create task state."""
    response = client.post("/v1/run", files={"config": (name, content)})
    assert response.status_code == code
    assert client.get("/v1/status").json()["occupancy"] == "idle"


def test_uploaded_name_and_content_preserved(client, monkeypatch, tmp_path):
    """The original scenario stem remains available for SUMO asset selection."""
    monkeypatch.setattr(main, "TASKS_DIR", tmp_path / "uploads")
    spawn = AsyncMock(return_value=Mock(pid=os.getpid()))
    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", spawn)
    response = client.post("/v1/run", files={"config": ("../demo.yaml", b"a: 1")})
    assert response.status_code == 202
    saved = tmp_path / "uploads" / response.json()["task_id"] / "demo.yaml"
    assert saved.read_bytes() == b"a: 1"
    assert str(saved) in spawn.call_args.args


def test_identity_recovery_and_tombstone(tmp_path):
    """Installation identity survives restart; uncertain tasks remain occupied and old tokens cannot revoke replacements."""
    store = Store(tmp_path)
    runner_id = store.runner_id
    store.configure(str(uuid4()), "old", "http://manager", "http://runner")
    revoked = store.revoke("old")
    assert revoked
    store.close()
    store = Store(tmp_path)
    assert store.runner_id == runner_id
    assert store.authorized("old", tombstone=True) and not store.authorized("old")
    store.configure(str(uuid4()), "new", "http://manager", "http://runner")
    revoked = store.revoke("old")
    assert revoked and store.authorized("new")
    with store.db:
        store.db.execute("INSERT INTO tasks(id, state, active) VALUES ('lost', 'RUNNING', 1)")
    runner = Runner(store)
    snapshot = runner.snapshot()
    assert snapshot["occupancy"] == "unknown" and snapshot["current_task"]["id"] == "lost"
    revoked = store.revoke("new")
    assert not revoked
    assert (tmp_path / "runner.sqlite3").stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    store.close()


def test_enrollment_retries_same_identity(tmp_path, monkeypatch):
    """Transient enrollment failure preserves credentials and retries the exact binding."""
    import httpx
    from api import runner as module

    async def exercise():
        store = Store(tmp_path)
        store.configure(str(uuid4()), "secret", "http://manager", "http://runner")
        runner = Runner(store)
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(503 if len(requests) == 1 else 200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client)
        monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
        await runner.enroll()
        assert len(requests) == 2
        assert requests[0].content == requests[1].content
        assert requests[0].headers["Authorization"] == "Bearer secret"
        store.close()

    asyncio.run(exercise())


def test_independent_recorder_survives_api_owner(tmp_path):
    """A separate recorder writes the result after its original API connection closes."""
    import subprocess
    import sys

    store = Store(tmp_path)
    with store.db:
        store.db.execute("INSERT INTO tasks(id, state, active) VALUES ('job', 'PREPARING', 1)")
    script = (
        "from pathlib import Path; from api import supervisor; import sys; "
        "supervisor.components_idle = lambda: True; "
        "supervisor.supervise(Path(sys.argv[1]), 'job', [sys.executable, '-c', 'import time; time.sleep(0.1)'])"
    )
    recorder = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    store.close()
    try:
        assert recorder.wait(timeout=5) == 0
        restored = Store(tmp_path)
        result = restored.db.execute("SELECT state, active FROM tasks WHERE id = 'job'").fetchone()
        assert tuple(result) == ("SUCCEEDED", 0)
        restored.close()
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


def test_cancellation_waits_for_component_cleanup(tmp_path, monkeypatch):
    """A cancelled child does not release occupancy until uncertain adapter cleanup becomes confirmed."""
    from api import supervisor

    store = Store(tmp_path)
    with store.db:
        store.db.execute("INSERT INTO tasks(id, state, active, cancel_requested) VALUES ('cancel', 'PREPARING', 1, 1)")
    child = Mock(pid=12345, returncode=0)
    child.poll.side_effect = [None, 0]
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: child)
    signal_process = Mock()
    monkeypatch.setattr(supervisor.os, "killpg", signal_process)
    monkeypatch.setattr(supervisor, "group_has_live_members", lambda group: False)
    cleanup = iter([False, True])
    monkeypatch.setattr(supervisor, "components_idle", lambda: next(cleanup))

    def pause(delay):
        if delay == 2:
            row = store.db.execute("SELECT state, active FROM tasks WHERE id = 'cancel'").fetchone()
            assert tuple(row) == ("UNKNOWN", 1)

    monkeypatch.setattr(supervisor.time, "sleep", pause)
    supervisor.supervise(tmp_path, "cancel", ["fake-scenario"])
    row = store.db.execute("SELECT state, active FROM tasks WHERE id = 'cancel'").fetchone()
    assert tuple(row) == ("CANCELLED", 0)
    signal_process.assert_called_once_with(12345, supervisor.signal.SIGINT)
    store.close()


def test_status_remains_available_during_stop(tmp_path, monkeypatch):
    """A slow cancellation wait leaves status responsive and returns only after durable completion."""
    import httpx

    async def exercise():
        store = Store(tmp_path)
        store.configure(str(uuid4()), "secret", "http://manager", "http://runner")
        with store.db:
            store.db.execute("INSERT INTO tasks VALUES ('running', 'RUNNING', 1, ?, ?, 0)", (os.getpid(), process_identity(os.getpid())))
        runner = Runner(store)
        monkeypatch.setattr(main, "runner", runner)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://runner", headers={"Authorization": "Bearer secret"}
        ) as client:
            stopping = asyncio.create_task(client.post("/v1/stop"))
            try:
                while not store.db.execute("SELECT cancel_requested FROM tasks").fetchone()[0]:
                    await asyncio.sleep(0)
                response = await asyncio.wait_for(client.get("/v1/status"), timeout=1)
                assert response.json()["occupancy"] == "busy"
                with store.db:
                    store.db.execute("UPDATE tasks SET active = 0, state = 'CANCELLED'")
                assert (await stopping).json()["occupancy"] == "idle"
            finally:
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
        store.close()

    asyncio.run(exercise())


def test_cleanup_retries_only_retained_components(tmp_path, monkeypatch):
    """A transient stop failure retains ownership and a later attempt completes cleanup without touching other adapters."""
    import io
    from api import supervisor

    store = Store(tmp_path)
    with store.db:
        store.db.execute("INSERT INTO task_components VALUES ('job', 'http://owned')")
    calls = []
    stopping = False
    fail_stop = True

    def request(url, timeout):
        nonlocal stopping, fail_stop
        address = url if isinstance(url, str) else url.full_url
        calls.append(address)
        if address.endswith("/stop"):
            if fail_stop:
                fail_stop = False
                raise OSError("temporary outage")
            stopping = True
            return io.BytesIO(b"{}")
        return io.BytesIO(b'{"state":"STOPPED"}' if stopping else b'{"state":"RUNNING"}')

    monkeypatch.setattr(supervisor, "urlopen", request)
    complete = supervisor.retry_component_cleanup(store, "job")
    assert not complete
    assert store.db.execute("SELECT count(*) FROM task_components").fetchone()[0] == 1
    complete = supervisor.retry_component_cleanup(store, "job")
    assert complete
    assert store.db.execute("SELECT count(*) FROM task_components").fetchone()[0] == 0
    assert calls.count("http://owned/v1/stop") == 2
    assert all(address.startswith("http://owned/") for address in calls)
    store.close()
