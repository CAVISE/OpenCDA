import logging
from typing import Any, Mapping

import opencda.metrics_tools.metrics  # noqa: F401
from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricCollection, MetricIssue, MetricSeries
from opencda.metrics_tools.registry import MetricRegistry

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metric_collector")


class MetricCollector:
    """
    Universal runtime collector for module reports.

    Parameters
    ----------
    module : str
        Module name the collector is responsible for.
    entity_id : int | str
        Identifier of the runtime entity being measured.
    metric_configs : Mapping[str, Mapping[str, Any]] | None
        Unified metric configuration. The keys define enabled metrics and
        the values provide per-metric constructor parameters. If omitted,
        all registered metrics are enabled with default parameters.
    """

    def __init__(
        self,
        module: str,
        entity_id: int | str,
        metric_configs: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.module = module
        self.entity_id = entity_id
        self.metric_configs = {metric_name: dict(params) for metric_name, params in (metric_configs or {}).items()}

        available_metrics = MetricRegistry.list_metrics()
        requested_metrics = self._resolve_requested_metrics(metric_configs, available_metrics)

        self.metrics: dict[str, BaseMetric] = {}
        self.disabled_metrics: list[str] = [name for name in available_metrics if name not in requested_metrics]
        self.unsupported_metrics: dict[str, str] = {}

        self._initialize_metrics(requested_metrics)
        logger.info(
            "Initialized metric collector module=%s entity_id=%s active=%s disabled=%s unsupported=%s",
            self.module,
            self.entity_id,
            self.active_metrics,
            self.disabled_metrics,
            sorted(self.unsupported_metrics),
        )

    @property
    def active_metrics(self) -> list[str]:
        """Return the names of currently active metric instances."""
        return list(self.metrics)

    def _resolve_requested_metrics(
        self,
        metric_configs: Mapping[str, Mapping[str, Any]] | None,
        available_metrics: list[str],
    ) -> list[str]:
        if metric_configs is not None:
            return list(dict.fromkeys(metric_configs))
        return list(available_metrics)

    def _initialize_metrics(self, requested_metrics: list[str]) -> None:
        for metric_name in requested_metrics:
            try:
                metric_cls = MetricRegistry.get_metric_class(metric_name=metric_name)
            except KeyError:
                self.unsupported_metrics[metric_name] = f"Metric '{metric_name}' is not registered."
                logger.warning(
                    "Skipping unknown metric module=%s entity_id=%s metric=%s",
                    self.module,
                    self.entity_id,
                    metric_name,
                )
                continue

            metric = metric_cls(**self.metric_configs.get(metric_name, {}))
            self.metrics[metric_name] = metric
            logger.info(
                "Activated metric module=%s entity_id=%s metric=%s params=%s",
                self.module,
                self.entity_id,
                metric_name,
                self.metric_configs.get(metric_name, {}),
            )

    def update(self, context: Mapping[str, Any]) -> None:
        """Update all active metrics from the provided context."""
        for metric in self.metrics.values():
            metric.update(context)

    def get_metric(self, metric_name: str) -> BaseMetric | None:  # noqa DC04
        """Return an active metric instance by name."""
        return self.metrics.get(metric_name)

    def get_raw(self) -> MetricCollection:
        """
        Return normalized raw metric data for reporting.
        """
        series: list[MetricSeries] = []
        for metric in self.metrics.values():
            series.extend(metric.get_raw())

        logger.info(
            "Exporting raw metrics module=%s entity_id=%s series=%d",
            self.module,
            self.entity_id,
            len(series),
        )
        return MetricCollection(
            module=self.module,
            entity_id=self.entity_id,
            active_metrics=tuple(self.active_metrics),
            disabled_metrics=tuple(self.disabled_metrics),
            unsupported_metrics=tuple(
                MetricIssue(metric_name=metric_name, reason=reason) for metric_name, reason in self.unsupported_metrics.items()
            ),
            series=tuple(series),
        )
