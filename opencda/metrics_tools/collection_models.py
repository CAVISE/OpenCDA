"""Raw collection models used between metrics, collectors, and reports."""

from dataclasses import asdict, dataclass

from opencda.metrics_tools.metric_sample import MetricSample


@dataclass(frozen=True, slots=True)
class MetricIssue:
    """Represents a metric that could not be activated."""

    metric_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """A named sequence of raw metric samples."""

    name: str
    samples: tuple[MetricSample, ...]


@dataclass(frozen=True, slots=True)
class MetricCollection:
    """Normalized raw metrics collected for a single runtime entity."""

    module: str
    entity_id: int | str
    active_metrics: tuple[str, ...]
    disabled_metrics: tuple[str, ...]
    unsupported_metrics: tuple[MetricIssue, ...]
    series: tuple[MetricSeries, ...]

    def get_series(self, series_name: str) -> tuple[MetricSample, ...]:
        """Return the samples for a named series."""
        for series in self.series:
            if series.name == series_name:
                return series.samples
        return ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the collection."""
        return asdict(self)
