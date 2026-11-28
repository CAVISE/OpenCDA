"""Authenticated enrollment and on-demand status snapshots."""

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import HTTPException
from api.state import Store

logger = logging.getLogger("opencda.runner")


class Runner:
    """Serialize registration and execution changes on the API event loop."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.instance_id = str(uuid4())
        self.revision = 0
        self.last: dict[str, Any] | None = None
        self.lock = asyncio.Lock()
        self.closed = False

    def snapshot(self) -> dict[str, Any]:
        """Capture a complete observation and assign a revision only when it changes."""
        self.store.recover()
        binding = self.store.binding()
        task = self.store.active_task()
        snapshot = {
            "registration_id": binding["registration_id"] if binding else "",
            "runner_id": self.store.runner_id,
            "instance_id": self.instance_id,
            "api_version": "v1",
            "version": "0.1.0",
            "capabilities": [],
            "ready": binding is not None and (task is None or task["state"] != "UNKNOWN"),
            "occupancy": "idle" if task is None else "unknown" if task["state"] == "UNKNOWN" else "busy",
            "current_task": {"id": task["id"], "state": task["state"]} if task else None,
        }
        if snapshot != self.last:
            self.revision += 1
            self.last = snapshot
        return dict(snapshot, revision=self.revision)

    async def enroll(self) -> None:
        """Retry transient enrollment failures while the reverse status endpoint is live."""
        delay = 1.0
        async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
            while not self.closed:
                binding = self.store.binding()
                if binding is None:
                    return
                try:
                    response = await client.post(
                        binding["manager_url"] + "/api/v1/runners/" + binding["registration_id"] + "/register",
                        headers={"Authorization": "Bearer " + binding["token"]},
                        json={"runner_id": self.store.runner_id, "api_url": binding["api_url"]},
                    )
                    if response.status_code == 200:
                        return
                    if response.status_code < 500 and response.status_code not in (408, 429):
                        logger.error("Enrollment rejected (HTTP %s); check registration configuration and restart API", response.status_code)
                        return
                except httpx.HTTPError:
                    # Transport failures retry the same persisted enrollment after backoff.
                    pass
                logger.warning("Manager enrollment unavailable; retrying")
                await asyncio.sleep(random.uniform(delay / 2, delay))
                delay = min(delay * 2, 30)


def configured_runner() -> Runner:
    """Load persisted configuration, accepting a secret file or initial environment token."""
    directory = Path(os.environ.get("CAVISE_RUNNER_STATE_DIR", ".runner-state"))
    store = Store(directory)
    try:
        registration = os.environ.get("CAVISE_REGISTRATION_ID")
        if registration:
            registration = str(UUID(registration))
            token_file = os.environ.get("CAVISE_REGISTRATION_TOKEN_FILE")
            token = Path(token_file).read_text().strip() if token_file else os.environ.get("CAVISE_REGISTRATION_TOKEN", "")
            existing = store.binding()
            if not token and existing and existing["registration_id"] == registration:
                token = existing["token"]
            manager_url = os.environ.get("CAVISE_MANAGER_URL", "").rstrip("/")
            api_url = os.environ.get("CAVISE_RUNNER_API_URL", "").rstrip("/")
            for address in (manager_url, api_url):
                parsed = httpx.URL(address)
                if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo or parsed.query or parsed.fragment:
                    raise ValueError("Configure direct HTTP(S) Manager and Runner base URLs without credentials, query or fragment")
            if not token:
                raise ValueError("Registration credential is required")
            # Deployment environment can still contain the retired bootstrap
            # credential after a successful unbind. Keep serving its tombstone
            # after restart without recreating the retired binding.
            if existing is None and store.authorized(token, tombstone=True):
                logger.info("Bootstrap credential was retired; awaiting new enrollment configuration")
            else:
                store.configure(registration, token, manager_url, api_url)
        return Runner(store)
    except Exception:
        store.close()
        raise


def credential(authorization: str) -> str:
    """Accept bearer credentials without exposing their contents in errors."""
    return authorization[7:] if authorization.startswith("Bearer ") else ""


def require_binding(runner: Runner, token: str) -> None:
    """Recheck credentials under the lifecycle lock before accepting execution."""
    if not runner.store.authorized(token):
        raise HTTPException(401, "Invalid registration credential")
