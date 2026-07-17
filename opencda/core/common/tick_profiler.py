"""Lightweight opt-in timing for one simulation tick."""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

logger = logging.getLogger("cavise.opencda.opencda.core.common.tick_profiler")


@dataclass(slots=True)
class _SectionTiming:
    total_seconds: float = 0.0
    calls: int = 0


class TickProfiler:
    """Aggregate named timings and emit a compact report after each tick."""

    _SCENARIO_SECTIONS = (
        "sumo_tick",
        "carla_tick",
        "world_frame",
        "spectator",
        "platoons",
        "cav_update",
        "rsu_update",
        "coperception",
        "capi_exchange",
        "behavior_services",
        "finish_step",
        "post_update",
    )
    _AGENT_SECTIONS = (
        "localization",
        "perception",
        "map",
        "safety",
        "behavior",
        "control",
    )

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._sections: defaultdict[str, _SectionTiming] = defaultdict(_SectionTiming)
        self._tick_number: int | None = None
        self._tick_started_at: float | None = None

    def start_tick(self, tick_number: int) -> None:
        """Reset accumulated values and start measuring a new tick."""
        if not self.enabled:
            return
        self._sections.clear()
        self._tick_number = tick_number
        self._tick_started_at = perf_counter()

    @contextmanager
    def measure(self, section: str) -> Iterator[None]:
        """Measure one invocation of a named section."""
        if not self.enabled:
            yield
            return

        started_at = perf_counter()
        try:
            yield
        finally:
            timing = self._sections[section]
            timing.total_seconds += perf_counter() - started_at
            timing.calls += 1

    def finish_tick(self) -> None:
        """Log the current tick report."""
        if not self.enabled or self._tick_started_at is None or self._tick_number is None:
            return

        total_seconds = perf_counter() - self._tick_started_at
        logger.info(
            "[tick-timing] tick=%d total=%.3fs | %s",
            self._tick_number,
            total_seconds,
            self._format_sections(self._SCENARIO_SECTIONS, include_average=False),
        )
        logger.info(
            "[tick-timing] tick=%d agents | %s",
            self._tick_number,
            self._format_sections(self._AGENT_SECTIONS, include_average=True),
        )

    def _format_sections(self, names: tuple[str, ...], include_average: bool) -> str:
        values: list[str] = []
        for name in names:
            timing = self._sections.get(name)
            if timing is None:
                continue

            total_ms = timing.total_seconds * 1000.0
            value = f"{name}={total_ms:.1f}ms"
            if include_average:
                average_ms = total_ms / timing.calls
                value += f" ({timing.calls}x, avg={average_ms:.3f}ms)"
            values.append(value)
        return " | ".join(values) if values else "no measured sections"
