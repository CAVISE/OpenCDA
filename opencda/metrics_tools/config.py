"""Helpers for resolving metric collector configuration from module configs."""

from typing import Any, Mapping


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
        enabled_metrics = list(enabled_metrics)

    metric_params = {
        metric_name: dict(params)
        for metric_name, params in (default_metric_params or {}).items()
    }
    for metric_name, params in metrics_config.get("metric_params", {}).items():
        merged_params = dict(metric_params.get(metric_name, {}))
        merged_params.update(dict(params))
        metric_params[metric_name] = merged_params

    return enabled_metrics, metric_params
