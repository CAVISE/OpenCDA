"""External, authenticated OpenCDA Runner API."""

import asyncio
import logging
import os
import sqlite3
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer

from api.models import Snapshot, Task
from api.runner import Runner, configured_runner, credential, require_binding
from api.state import process_identity

logger = logging.getLogger("opencda.api")
TASKS_DIR = Path(__file__).resolve().parents[1] / "runner_tasks"
MAX_CONFIG_SIZE = 1024 * 1024
runner: Runner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own enrollment resources; independent execution recorders survive API shutdown."""
    global runner
    runner = configured_runner()
    enrollment = asyncio.create_task(runner.enroll())
    try:
        yield
    finally:
        runner.closed = True
        enrollment.cancel()
        await asyncio.gather(enrollment, return_exceptions=True)
        runner.store.close()
        runner = None


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/v1", dependencies=[Depends(HTTPBearer(auto_error=False))])


def runtime() -> Runner:
    """Reject requests before application startup."""
    if runner is None:
        raise HTTPException(503, "Runner is starting")
    return runner


@app.exception_handler(HTTPException)  # noqa: DC02 - FastAPI invokes this registered handler.
async def http_error(request: Request, error: HTTPException) -> JSONResponse:
    """Return shared JSON errors without internal exception details."""
    return JSONResponse(status_code=error.status_code, content={"code": f"http_{error.status_code}", "message": str(error.detail)})


@app.exception_handler(RequestValidationError)  # noqa: DC02 - FastAPI invokes this registered handler.
async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """Do not echo request contents in validation failures."""
    return JSONResponse(status_code=422, content={"code": "invalid_request", "message": "Invalid request parameters."})


@app.middleware("http")  # noqa: DC02 - FastAPI invokes this registered handler.
async def authenticate(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Protect all external routes, including documentation and configuration upload."""
    if request.url.path == "/healthz":
        return await call_next(request)
    token = credential(request.headers.get("Authorization", ""))
    tombstone = request.method == "DELETE" and request.url.path == "/v1/registration"
    if runner is None or not runner.store.authorized(token, tombstone=tombstone):
        return JSONResponse(status_code=401, content={"code": "unauthorized", "message": "Invalid registration credential."})
    try:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
    except (OSError, sqlite3.Error):
        logger.error("Runner operation failed")
        return JSONResponse(status_code=500, content={"code": "internal_error", "message": "Runner operation failed."})


@app.get("/healthz")  # noqa: DC02 - FastAPI invokes this registered handler.
async def health() -> dict[str, str]:
    """Report liveness without revealing identity or credentials."""
    return {"status": "ok"}


@router.get("/status", response_model=Snapshot)
async def status() -> dict[str, Any]:
    """Capture state without waiting for scenario execution or cleanup."""
    return runtime().snapshot()


@router.get("/tasks/{task_id}", response_model=Task)  # noqa: DC02 - FastAPI invokes this registered handler.
async def task_status(task_id: str) -> dict[str, str]:
    """Return retained task observations, including across process restarts."""
    row = runtime().store.db.execute("SELECT id, state FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Task result is not known")
    return dict(row)


@router.delete("/registration", status_code=204)  # noqa: DC02 - FastAPI invokes this registered handler.
async def unregister(request: Request) -> Response:
    """Serialize freedom checks and revocation with acceptance of new work."""
    current = runtime()
    token = credential(request.headers.get("Authorization", ""))
    async with current.lock:
        if not current.store.revoke(token):
            raise HTTPException(409, "Runner freedom could not be confirmed")
    return Response(status_code=204)


async def _start_scenario(scenario: str, task_id: str) -> dict[str, str | int]:
    """Reserve occupancy durably before launching the independent recorder."""
    current = runtime()
    with current.store.db:
        current.store.db.execute(
            "INSERT INTO tasks(id, state, active, supervisor_pid, supervisor_identity) VALUES (?, 'PREPARING', 1, ?, ?)",
            (task_id, os.getpid(), process_identity(os.getpid())),
        )
    command = [
        sys.executable,
        "opencda.py",
        "-t",
        scenario,
        "-c",
        "--manage-components",
        "--carla-adapter-url",
        os.environ.get("CAVISE_CARLA_API_URL", "http://carla:8080"),
        "--sumo-adapter-url",
        os.environ.get("CAVISE_SUMO_API_URL", "http://sumo:8080"),
    ]
    try:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "api.supervisor",
            str(current.store.directory),
            task_id,
            *command,
            start_new_session=True,
        )
    except OSError:
        with current.store.db:
            current.store.db.execute("UPDATE tasks SET state = 'FAILED', active = 0 WHERE id = ?", (task_id,))
        raise
    with current.store.db:
        current.store.db.execute(
            "UPDATE tasks SET supervisor_pid = ?, supervisor_identity = ? WHERE id = ? AND supervisor_pid = ?",
            (child.pid, process_identity(child.pid), task_id, os.getpid()),
        )
    current.snapshot()
    return {"task_id": task_id, "message": "Scenario preparation accepted", "pid": child.pid}


@router.post("/run", status_code=202)
async def run(request: Request, scenario: str | None = None, config: UploadFile | None = None) -> dict[str, str | int]:
    """Reject duplicate or uncertain execution, preserving named and uploaded scenarios."""
    if (scenario is None) == (config is None):
        raise HTTPException(422, "Provide either scenario or config, but not both")
    if scenario is not None and not scenario.strip():
        raise HTTPException(422, "Scenario must not be empty")
    current = runtime()
    async with current.lock:
        require_binding(current, credential(request.headers.get("Authorization", "")))
        if current.store.active_task() is not None:
            raise HTTPException(409, "A scenario is running or unresolved")
        if config is not None:
            return await _run_from_file(config)
        assert scenario is not None
        return await _run_from_name(scenario)


@router.post("/stop", response_model=Snapshot)
async def stop(request: Request) -> dict[str, Any]:
    """Persist cancellation intent while leaving state reads responsive."""
    current = runtime()
    async with current.lock:
        require_binding(current, credential(request.headers.get("Authorization", "")))
        task = current.store.active_task()
        if task is None:
            return current.snapshot()
        with current.store.db:
            current.store.db.execute("UPDATE tasks SET cancel_requested = 1 WHERE id = ?", (task["id"],))
    deadline = asyncio.get_running_loop().time() + 30
    while current.store.db.execute("SELECT active FROM tasks WHERE id = ?", (task["id"],)).fetchone()[0]:
        if asyncio.get_running_loop().time() >= deadline:
            raise HTTPException(504, "Scenario stop or cleanup remains unconfirmed")
        await asyncio.sleep(0.1)
    return current.snapshot()


async def _run_from_name(scenario: str) -> dict[str, str | int]:
    """Start a named scenario while the caller holds the process lock."""
    return {"status": "STARTING", **await _start_scenario(scenario, str(uuid4()))}


async def _run_from_file(config: UploadFile) -> dict[str, str | int]:
    """Save and start an uploaded YAML while the caller holds the process lock."""
    content = await config.read(MAX_CONFIG_SIZE + 1)
    if len(content) > MAX_CONFIG_SIZE:
        raise HTTPException(413, "Configuration is too large")

    try:
        parsed = yaml.safe_load(content)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise HTTPException(422, "Invalid YAML configuration") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, "Configuration must be a mapping")

    filename = Path((config.filename or "scenario.yaml").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".yaml":
        raise HTTPException(422, "Configuration filename must end in .yaml")
    # The stem also selects the existing SUMO scenario assets.
    if not Path(filename).stem:
        raise HTTPException(422, "Configuration filename must include a scenario name")

    task_id = uuid4()
    task_dir = TASKS_DIR.resolve() / str(task_id)
    task_dir.mkdir(parents=True)
    config_path = task_dir / filename
    config_path.write_bytes(content)
    logger.info("Configuration saved: task_id=%s path=%s", task_id, config_path)
    result = await _start_scenario(str(config_path), str(task_id))
    return {"status": "STARTING", **result}


app.include_router(router)
