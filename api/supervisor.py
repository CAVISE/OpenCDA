"""Record scenario completion independently of the HTTP API process."""

import os
import signal
import subprocess
import sys
import time
import json
from pathlib import Path
from urllib.request import Request, urlopen

from api.state import Store, process_identity


def components_idle() -> bool:
    """Require confirmed cleanup of both internal adapters before releasing occupancy."""
    for name, default in (("CARLA", "http://carla:8080"), ("SUMO", "http://sumo:8080")):
        try:
            with urlopen(os.environ.get(f"CAVISE_{name}_API_URL", default) + "/v1/status", timeout=5) as response:
                if json.load(response).get("state") not in {"IDLE", "STOPPED", "FAILED"}:
                    return False
        except (OSError, ValueError):
            return False
    return True


def retry_component_cleanup(store: Store, task_id: str) -> bool:
    """Retry only this task's recorded adapters, retaining every ambiguous stop."""
    pending = store.db.execute("SELECT base_url FROM task_components WHERE task_id = ? ORDER BY rowid DESC", (task_id,)).fetchall()
    complete = True
    for row in pending:
        base = row["base_url"]
        try:
            with urlopen(base + "/v1/status", timeout=5) as response:
                state = json.load(response).get("state")
            if state not in {"IDLE", "STOPPED", "FAILED"}:
                with urlopen(Request(base + "/v1/stop", method="POST"), timeout=40) as response:
                    response.read()
                with urlopen(base + "/v1/status", timeout=5) as response:
                    state = json.load(response).get("state")
            if state not in {"IDLE", "STOPPED", "FAILED"}:
                complete = False
                continue
            with store.db:
                store.db.execute("DELETE FROM task_components WHERE task_id = ? AND base_url = ?", (task_id, base))
        except (OSError, ValueError):
            # A lost stop acknowledgement leaves responsibility durable for the next retry.
            complete = False
    return complete


def group_has_live_members(group_id: int) -> bool:
    """Retain occupancy while descendants outlive the scenario launcher."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            # An unreadable process table cannot prove freedom.
            return True
    return False


def wait_for_child(store: Store, task_id: str, child: subprocess.Popen) -> bool:
    """Wait for child exit and report whether this supervisor delivered cancellation."""
    signalled = False
    while child.poll() is None:
        task = store.db.execute("SELECT cancel_requested FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task[0] and not signalled:
            try:
                os.killpg(child.pid, signal.SIGINT)
                signalled = True
            except ProcessLookupError:
                # The child exited before cancellation; reconcile its actual result below.
                pass
        time.sleep(0.1)

    return signalled


def supervise(directory: Path, task_id: str, command: list[str]) -> None:
    """Own the child and persist its result before releasing the active task slot."""
    store = Store(directory)
    with store.db:
        store.db.execute(
            "UPDATE tasks SET supervisor_pid = ?, supervisor_identity = ? WHERE id = ? AND active = 1",
            (os.getpid(), process_identity(os.getpid()), task_id),
        )
    environment = dict(os.environ, CAVISE_TASK_ID=task_id, CAVISE_RUNNER_STATE_DIR=str(directory))
    try:
        child = subprocess.Popen(command, env=environment, start_new_session=True)
    except OSError:
        with store.db:
            store.db.execute("UPDATE tasks SET state = 'FAILED', active = 0 WHERE id = ?", (task_id,))
        store.close()
        return

    signalled = wait_for_child(store, task_id, child)

    # A dead API cannot prevent this process from recording the outcome. Failed
    # adapter cleanup stays occupied and is retried without relaunching execution.
    while group_has_live_members(child.pid) or not retry_component_cleanup(store, task_id) or not components_idle():
        with store.db:
            store.db.execute("UPDATE tasks SET state = 'UNKNOWN' WHERE id = ?", (task_id,))
        time.sleep(2)
    outcome = "CANCELLED" if signalled and child.returncode in (0, -signal.SIGINT, 130) else "SUCCEEDED" if child.returncode == 0 else "FAILED"
    with store.db:
        store.db.execute("UPDATE tasks SET state = ?, active = 0 WHERE id = ?", (outcome, task_id))
    store.close()


def mark_running() -> None:
    """Called by the scenario after its first successful simulator tick."""
    task_id = os.environ.get("CAVISE_TASK_ID")
    if task_id:
        store = Store(Path(os.environ["CAVISE_RUNNER_STATE_DIR"]))
        with store.db:
            store.db.execute("UPDATE tasks SET state = 'RUNNING' WHERE id = ? AND state = 'PREPARING'", (task_id,))
        store.close()


if __name__ == "__main__":
    supervise(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
