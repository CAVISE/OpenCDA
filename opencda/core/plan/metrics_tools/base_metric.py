from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

from opencda.core.plan.metrics_tools.metric_sample import MetricSample
from opencda.core.plan.metrics_tools.registry import MetricRegistry
from opencda.core.plan.report_models import MetricSeries


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning of the simulation.
    sample_interval : float
        The time interval in seconds between metric updates.
    """

    metric_name: ClassVar[str]
    module: ClassVar[str] = "undefined-module"

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        MetricRegistry.register(cls)

    def __init__(self, warmup_steps: int = 100, sample_interval: float = 0.05):
        self.warmup_steps = warmup_steps
        self.sample_interval = sample_interval
        self.count = 0

    @classmethod
    def supports(cls, capabilities: Mapping[str, Any] | None = None) -> bool:
        """
        Check whether the metric can run with the provided capabilities.

        Parameters
        ----------
        capabilities : Mapping[str, Any] | None
            Optional capability flags for the current runtime.
        """
        return True

    def update(self, context: Mapping[str, Any]) -> None:
        """
        Update metric state from a normalized runtime context.

        Parameters
        ----------
        context : Mapping[str, Any]
            Runtime context containing the data required by the metric.
        """
        self.count += 1
        if self.count <= self.warmup_steps:
            return
        self._process_context(context)

    def _make_sample(self, value: float) -> MetricSample:
        """Create a scalar sample using the current tick and sample interval."""
        return MetricSample(
            tick=self.count,
            value=float(value),
        )

    @abstractmethod
    def _process_context(self, context: Mapping[str, Any]) -> None:
        """Process a context after the warmup period."""

    @abstractmethod
    def get_raw(self) -> tuple[MetricSeries, ...]:
        """Return collected metric series in normalized raw form."""
