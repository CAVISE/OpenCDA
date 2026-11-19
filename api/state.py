"""Transactional Runner identity, credentials and execution observations."""

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path
from uuid import uuid4


def process_identity(pid: int) -> str:
    """Identify a Linux process using boot ID and start ticks, never PID alone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return ""
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
    except (OSError, IndexError):
        return ""


class Store:
    """Own one connection; API and independent supervisor use separate connections."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        self.directory = directory.resolve()
        path = directory / "runner.sqlite3"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.row_factory = sqlite3.Row  # noqa: DC05 - SQLite calls this factory when fetching rows.
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS installation (id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS binding (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                registration_id TEXT NOT NULL, token TEXT NOT NULL,
                manager_url TEXT NOT NULL, api_url TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tombstones (digest BLOB PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, state TEXT NOT NULL, active INTEGER NOT NULL,
                supervisor_pid INTEGER, supervisor_identity TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS task_components (
                task_id TEXT NOT NULL, base_url TEXT NOT NULL,
                PRIMARY KEY (task_id, base_url)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_task ON tasks(active) WHERE active = 1;
        """)
        with self.db:
            if self.db.execute("SELECT id FROM installation").fetchone() is None:
                self.db.execute("INSERT INTO installation VALUES (?)", (str(uuid4()),))
        self.runner_id = self.db.execute("SELECT id FROM installation").fetchone()[0]

    def binding(self) -> sqlite3.Row | None:
        """Read the currently configured enrollment, including its private credential."""
        return self.db.execute("SELECT * FROM binding").fetchone()

    def configure(self, registration_id: str, token: str, manager_url: str, api_url: str) -> None:
        """Persist pending enrollment; refuse replacement of an existing binding."""
        with self.db:
            current = self.binding()
            desired = (registration_id, token, manager_url, api_url)
            if current is not None and tuple(current)[1:] != desired:
                raise ValueError("Unbind the existing registration before changing configuration")
            digest = hashlib.sha256(token.encode()).digest()
            if self.db.execute("SELECT 1 FROM tombstones WHERE digest = ?", (digest,)).fetchone():
                raise ValueError("A retired credential cannot be reused")
            self.db.execute("INSERT OR IGNORE INTO binding VALUES (1, ?, ?, ?, ?)", desired)

    def authorized(self, token: str, tombstone: bool = False) -> bool:
        """Compare fixed-size hashes; retired tokens authorize only unbinding retries."""
        digest = hashlib.sha256(token.encode()).digest()
        binding = self.binding()
        expected = hashlib.sha256(binding["token"].encode()).digest() if binding else bytes(32)
        active = hmac.compare_digest(digest, expected) and binding is not None
        retired = False
        if tombstone:
            retired = any(hmac.compare_digest(digest, row[0]) for row in self.db.execute("SELECT digest FROM tombstones"))
        return bool(token) and (active or retired)

    def active_task(self) -> sqlite3.Row | None:
        """Return the sole occupied or uncertain execution."""
        return self.db.execute("SELECT * FROM tasks WHERE active = 1").fetchone()

    def revoke(self, token: str) -> bool:
        """Atomically check freedom and retire only the binding matched by this token."""
        digest = hashlib.sha256(token.encode()).digest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM tombstones WHERE digest = ?", (digest,)).fetchone():
                return True
            if not self.authorized(token) or self.active_task() is not None:
                return False
            self.db.execute("INSERT INTO tombstones VALUES (?)", (digest,))
            self.db.execute("DELETE FROM binding")
        return True

    def recover(self) -> None:
        """Keep lost supervisors uncertain; disappearance never implies completion."""
        with self.db:
            task = self.active_task()
            if task and (not task["supervisor_identity"] or process_identity(task["supervisor_pid"]) != task["supervisor_identity"]):
                self.db.execute("UPDATE tasks SET state = 'UNKNOWN' WHERE id = ? AND active = 1", (task["id"],))

    def close(self) -> None:
        """Release this owner's SQLite connection."""
        self.db.close()


def retain_component(base_url: str) -> None:
    """Persist cleanup responsibility before an adapter start can take effect."""
    task_id = os.environ.get("CAVISE_TASK_ID")
    if task_id is None:
        return
    store = Store(Path(os.environ["CAVISE_RUNNER_STATE_DIR"]))
    try:
        with store.db:
            store.db.execute("INSERT OR IGNORE INTO task_components VALUES (?, ?)", (task_id, base_url))
    finally:
        store.close()
