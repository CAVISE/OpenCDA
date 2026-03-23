"""Helpers for resolving metric collector configuration from module configs."""

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def resolve_metric_collector_config(
    module_config: Mapping[str, Any] | None,
    default_enabled_metrics: list[str] | None = None,
    default_metric_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[str] | None, dict[str, dict[str, Any]]]:
    """
    Resolve collector configuration from a module config.
    """
    metrics_config = dict((module_config or {}).get("metrics", {}))

    enabled_metrics = metrics_config.get("enabled_metrics", default_enabled_metrics)
    if enabled_metrics is not None:
        enabled_metrics = list(dict.fromkeys(enabled_metrics))

    metric_params = {
        metric_name: dict(params)
        for metric_name, params in (default_metric_params or {}).items()
    }
    for metric_name, params in metrics_config.get("metric_params", {}).items():
        merged_params = dict(metric_params.get(metric_name, {}))
        merged_params.update(dict(params))
        metric_params[metric_name] = merged_params

    if enabled_metrics is not None:
        unexpected_metric_params = [
            metric_name for metric_name in metric_params if metric_name not in enabled_metrics
        ]
        if unexpected_metric_params:
            logger.error(
                "Invalid metric config: metric_params declared for disabled metrics: %s",
                ", ".join(sorted(unexpected_metric_params)),
            )
            raise ValueError(
                "metric_params provided for metrics that are not enabled: "
                + ", ".join(sorted(unexpected_metric_params))
            )

    logger.debug(
        "Resolved metric config enabled_metrics=%s metric_params=%s",
        enabled_metrics,
        sorted(metric_params),
    )
    return enabled_metrics, metric_params
