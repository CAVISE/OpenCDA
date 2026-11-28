"""Manage dedicated simulator adapters around scenario execution."""

import signal
import logging
import json
from argparse import Namespace
from types import TracebackType
from typing import Literal, Self
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from api.state import retain_component

logger = logging.getLogger(__name__)


def adapter_request(url: str, method: str = "POST") -> str:
    """Call the prototype adapter and propagate HTTP or transport failures."""
    with urlopen(Request(url, method=method), timeout=40) as response:
        return response.read().decode()


class ManagedComponents:
    """Start dedicated components and clean them up after scenario finalization.

    Prototype adapters have no ownership tokens. Their status is checked before
    starting, but exclusive use must also be enforced operationally.
    """

    def __init__(self, options: Namespace) -> None:
        """Record adapter addresses and the components selected by the CLI."""
        self.options = options
        self.started: list[str] = []

    def __enter__(self) -> Self:
        """Prepare components, rolling back partial preparation on failure."""
        try:
            if self.options.with_capi:
                raise ValueError("Managed Artery startup is not supported yet")
            if self.options.cosim:
                self._start(self.options.sumo_adapter_url, {"scenario": self.options.test_scenario})
            self._start(self.options.carla_adapter_url)
            return self
        except BaseException:
            self._cleanup(suppress_errors=True)
            raise

    def _start(self, base: str, params: dict[str, str] | None = None) -> None:
        """Reject occupied adapters and track even ambiguous start attempts."""
        base = base.rstrip("/")
        status = adapter_request(f"{base}/v1/status", "GET")
        available = json.loads(status).get("state") in {"IDLE", "STOPPED", "FAILED"}
        if not available:
            raise RuntimeError(f"Component adapter is already occupied: {base}")
        retain_component(base)
        self.started.append(base)
        query = "?" + urlencode(params) if params else ""
        adapter_request(f"{base}/v1/run{query}")

    def _cleanup(self, suppress_errors: bool = False) -> None:
        """Attempt every stop, preserving any original scenario exception."""
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            errors = []
            failed = []
            # SUMO must remain available throughout scenario finalization.
            for base in reversed(self.started):
                try:
                    logger.info("Stopping component through %s/v1/stop", base)
                    adapter_request(f"{base}/v1/stop")
                except Exception as error:
                    logger.exception("Failed to stop component adapter %s", base)
                    errors.append(error)
                    failed.append(base)
            self.started = list(reversed(failed))
            if errors and not suppress_errors:
                raise RuntimeError("Component cleanup failed") from errors[0]
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Stop components only after the scenario has finalized."""
        self._cleanup(suppress_errors=exc_type is not None)
        return False
