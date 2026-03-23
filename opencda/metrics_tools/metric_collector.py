from typing import Any, Mapping

import opencda.metrics_tools.metrics  # noqa: F401
from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricCollection, MetricIssue, MetricSeries
from opencda.metrics_tools.registry import MetricRegistry


class MetricCollector:
    """
    Universal runtime collector for module reports.

    Parameters
    ----------
    module : str
        Module name the collector is responsible for.
    entity_id : int | str
        Identifier of the runtime entity being measured.
    enabled_metrics : list[str] | None
        Explicit list of metric names to enable.
    capabilities : Mapping[str, Any] | None
        Capability flags describing the current runtime.
    metric_params : Mapping[str, Mapping[str, Any]] | None
        Per-metric constructor parameters.
    """

    def __init__(
        self,
        module: str,
        entity_id: int | str,
        enabled_metrics: list[str] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        metric_params: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.module = module
        self.entity_id = entity_id
        self.capabilities = dict(capabilities or {})
        self.metric_params = {name: dict(params) for name, params in (metric_params or {}).items()}

        available_metrics = MetricRegistry.list_metrics()
        self._requested_metrics = self._resolve_requested_metrics(enabled_metrics, available_metrics)
        self._validate_metric_params()

        self.metrics: dict[str, BaseMetric] = {}
        self.active_metrics: list[str] = []
        self.disabled_metrics: list[str] = [name for name in available_metrics if name not in self._requested_metrics]
        self.unsupported_metrics: dict[str, str] = {}

        self._initialize_metrics()

    def _resolve_requested_metrics(
        self,
        enabled_metrics: list[str] | None,
        available_metrics: list[str],
    ) -> list[str]:
        if enabled_metrics is not None:
            return list(dict.fromkeys(enabled_metrics))
        return list(available_metrics)

    def _validate_metric_params(self) -> None:
        unexpected_metric_params = [
            metric_name for metric_name in self.metric_params if metric_name not in self._requested_metrics
        ]
        if unexpected_metric_params:
            raise ValueError(
                "metric_params provided for metrics that are not enabled: "
                + ", ".join(sorted(unexpected_metric_params))
            )

    def _initialize_metrics(self) -> None:
        for metric_name in self._requested_metrics:
            try:
                metric_cls = MetricRegistry.get_metric_class(metric_name=metric_name)
            except KeyError:
                self.unsupported_metrics[metric_name] = f"Metric '{metric_name}' is not registered."
                continue

            if not metric_cls.supports(self.capabilities):
                self.unsupported_metrics[metric_name] = "Metric is not supported by current capabilities."
                continue

            metric = metric_cls(**self.metric_params.get(metric_name, {}))
            self.metrics[metric_name] = metric
            self.active_metrics.append(metric_name)

    def update(self, context: Mapping[str, Any]) -> None:
        """Update all active metrics from the provided context."""
        for metric in self.metrics.values():
            metric.update(context)

    def get_metric(self, metric_name: str) -> BaseMetric | None:
        """Return an active metric instance by name."""
        return self.metrics.get(metric_name)

    def get_raw(self) -> MetricCollection:
        """
        Return normalized raw metric data for reporting.
        """
        series: list[MetricSeries] = []
        for metric in self.metrics.values():
            series.extend(metric.get_raw())

        return MetricCollection(
            module=self.module,
            entity_id=self.entity_id,
            active_metrics=tuple(self.active_metrics),
            disabled_metrics=tuple(self.disabled_metrics),
            unsupported_metrics=tuple(
                MetricIssue(metric_name=metric_name, reason=reason)
                for metric_name, reason in self.unsupported_metrics.items()
            ),
            series=tuple(series),
        )
