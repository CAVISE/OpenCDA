"""Generic report models for metric-based reporting."""

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from opencda.metrics_tools.collection_models import MetricCollection, MetricIssue, MetricSeries


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    """Structured summary for a single metric series."""

    name: str
    count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None


@dataclass(frozen=True, slots=True)
class MetricSummarySpec:
    """Descriptor for how a metric series should be summarized."""

    series_name: str | None = None
    display_name: str | None = None
    cutoff: float | None = None
    resolver: Callable[[Callable[[str], Sequence[float]]], Sequence[float]] | None = None


@dataclass(frozen=True, slots=True)
class MetricReportSpec:
    """Descriptor for how a metric should be represented in a report."""

    metric_name: str
    display_name: str | None = None
    series_names: tuple[str, ...] = ()
    summary_specs: tuple[MetricSummarySpec, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityReportInfo:
    """Common report metadata for one runtime entity."""

    module: str
    entity_id: int | str
    context_id: int | str | None = None
    active_metrics: tuple[str, ...] = ()
    disabled_metrics: tuple[str, ...] = ()
    unsupported_metrics: tuple[MetricIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Report block for one metric."""

    metric_name: str
    display_name: str | None = None
    summary: tuple[SeriesSummary, ...] = ()
    series: tuple[MetricSeries, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityReport:
    """Structured report for one runtime entity."""

    info: EntityReportInfo
    metrics: tuple[MetricReport, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModuleReport:
    """Generic structured report for all entities of one module."""

    module: str
    entities: tuple[EntityReport, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the module report."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntityMetricCollections:
    """Generic bundle of raw metric collections for one entity."""

    entity_id: int | str
    context_id: int | str | None = None
    collections: tuple[MetricCollection, ...] = ()

    def get_collection(self, module: str) -> MetricCollection | None:
        """Return the raw collection for a specific module if present."""
        for collection in self.collections:
            if collection.module == module:
                return collection
        return None


@dataclass(frozen=True, slots=True)
class GroupReport:
    """Generic structured report for a grouped evaluation context."""

    group_id: int | str
    entities: tuple[EntityReport, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)
