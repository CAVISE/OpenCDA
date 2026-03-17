from typing import Any, Mapping

import opencda.core.plan.metrics.dynamics_metric as _dynamics_metric
import opencda.core.plan.metrics.ttc_metric as _ttc_metric
from opencda.core.plan.metrics.base_metric import BaseMetric
from opencda.core.plan.metrics.registry import MetricRegistry
from opencda.core.plan.report_models import MetricCollection, MetricIssue, MetricSeries


class MetricCollector:
    """
    Universal runtime collector for module metrics.

    Parameters
    ----------
    module : str
        Module name the collector is responsible for.
    entity_id : int | str
        Identifier of the runtime entity being measured.
    enabled_metrics : list[str] | None
        Explicit list of metric names to enable. If omitted, all
        registered metrics for the module are enabled.
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

        available_metrics = MetricRegistry.list_metrics(module=module)
        self._requested_metrics = list(enabled_metrics) if enabled_metrics is not None else list(available_metrics)

        self.metrics: dict[str, BaseMetric] = {}
        self.active_metrics: list[str] = []
        self.disabled_metrics: list[str] = [name for name in available_metrics if name not in self._requested_metrics]
        self.unsupported_metrics: dict[str, str] = {}

        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        for metric_name in self._requested_metrics:
            try:
                metric_cls = MetricRegistry.get_metric_class(module=self.module, metric_name=metric_name)
            except KeyError:
                self.unsupported_metrics[metric_name] = f"Metric '{metric_name}' is not registered for module '{self.module}'."
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
