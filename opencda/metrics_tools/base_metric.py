"""Base abstractions for runtime metric implementations."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec
from opencda.metrics_tools.registry import MetricRegistry


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
    metric_display_name: ClassVar[str | None] = None
    metric_series_name: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        MetricRegistry.register(cls)

    def __init__(self, warmup_steps: int = 100, sample_interval: float = 0.05):
        self.warmup_steps = warmup_steps
        self.sample_interval = sample_interval
        self.steps_count = 0

    def update(self, context: Mapping[str, Any]) -> None:
        """
        Update metric state from a normalized runtime context.

        Parameters
        ----------
        context : Mapping[str, Any]
            Runtime context containing the data required by the metric.
        """
        self.steps_count += 1
        if self.steps_count <= self.warmup_steps:
            return
        self._process_context(context)

    def _make_sample(self, value: float) -> MetricSample:
        """Create a scalar sample using the current tick and sample interval."""
        return MetricSample(
            tick=self.steps_count,
            value=float(value),
        )

    @abstractmethod
    def _process_context(self, context: Mapping[str, Any]) -> None:
        """Process a context after the warmup period."""
        raise NotImplementedError

    def get_raw(self) -> tuple[MetricSeries, ...]:
        """Return collected metric series in normalized raw form."""
        return (
            MetricSeries(
                name=self.metric_series_name or self.metric_name,
                samples=tuple(self._get_metric_samples()),
            ),
        )

    def _get_metric_samples(self) -> list[MetricSample]:
        samples = getattr(self, "_samples", None)
        if samples is None:
            raise NotImplementedError("Default get_raw() requires metric samples to be stored in self._samples.")
        return samples

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        """Return the report representation owned by the metric."""
        display_name = cls.metric_display_name or _format_metric_display_name(cls.metric_name)
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name=display_name,
            series_names=(cls.metric_name,),
            summary_specs=(MetricSummarySpec(series_name=cls.metric_name, display_name=display_name),),
        )


def _format_metric_display_name(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()
