import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
import signal
import sys
from uuid import uuid4

import yaml
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile
from fastapi import Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("opencda.api")

API_VERSION = "v1"

TASKS_DIR = Path(__file__).resolve().parents[1] / "runner_tasks"
MAX_CONFIG_SIZE = 1024 * 1024

app = FastAPI()
router = APIRouter(prefix=f"/{API_VERSION}")
process = None
process_lock = asyncio.Lock()
stop_requested = False
last_pid = None
last_returncode = None


async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Report expected rejections and translate operating-system failures."""
    try:
        response = await call_next(request)
    except OSError:
        logger.exception("%s %s failed: pid=%s", request.method, request.url.path, process.pid if process else None)
        return JSONResponse(status_code=500, content={"detail": "OpenCDA operation failed"})
    if response.status_code >= 400:
        logger.warning("%s %s rejected: HTTP %s", request.method, request.url.path, response.status_code)
    elif request.url.path.endswith("/status"):
        logger.debug("Status requested: HTTP %s", response.status_code)
    return response


app.middleware("http")(log_requests)


@router.get("/status")
async def status() -> dict[str, str | int | None]:
    if process is None:
        state = "IDLE" if last_pid is None else "STOPPED"
        return {"state": state, "pid": last_pid, "returncode": last_returncode}
    if process.returncode is None:
        state = "STOPPING" if stop_requested else "RUNNING"
    else:
        state = "STOPPED" if process.returncode == 0 else "FAILED"
    return {"state": state, "pid": process.pid, "returncode": process.returncode}


async def _start_scenario(scenario: str, task_id: str) -> dict[str, str | int]:
    """Start a scenario while the caller holds the process lock."""
    global process, stop_requested
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "opencda.py",
        "-t",
        scenario,
        "-c",
        "--manage-components",
        start_new_session=True,
    )
    stop_requested = False
    logger.info("Scenario started: scenario=%s task_id=%s pid=%s", scenario, task_id, process.pid)
    return {"task_id": task_id, "message": f"Start scenario: {scenario}. Process {process}", "pid": process.pid}


@router.post("/run", status_code=202)
async def run(scenario: str | None = None, config: UploadFile | None = None) -> dict[str, str | int]:
    """Start exactly one named scenario or uploaded configuration."""
    logger.info("Run requested: scenario=%s upload=%s", scenario, config is not None)
    if (scenario is None) == (config is None):
        raise HTTPException(422, "Provide either scenario or config, but not both")
    if scenario is not None and not scenario.strip():
        raise HTTPException(422, "Scenario must not be empty")
    async with process_lock:
        if process is not None and process.returncode is None:
            raise HTTPException(409, "A scenario is already running")
        if config is not None:
            return await _run_from_file(config)
        assert scenario is not None
        return await _run_from_name(scenario)


@router.post("/stop")
async def stop() -> dict[str, str | int | None]:
    global process, stop_requested, last_pid, last_returncode
    logger.info("Stop requested")
    async with process_lock:
        if process is not None:
            current = process
            if process.returncode is None and not stop_requested:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    logger.info("Process already exited before SIGINT: pid=%s", process.pid)
                stop_requested = True

            try:
                await asyncio.wait_for(process.wait(), timeout=30)
                logger.info("Process stopped: pid=%s returncode=%s", process.pid, process.returncode)
                last_pid = process.pid
                last_returncode = process.returncode
                process = None
                stop_requested = False
            except asyncio.TimeoutError:
                logger.error("Stop timed out: pid=%s", current.pid)
                raise HTTPException(status_code=504, detail="OpenCDA is still stopping")
        return {"state": "IDLE" if last_pid is None else "STOPPED", "pid": last_pid, "returncode": last_returncode}


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
