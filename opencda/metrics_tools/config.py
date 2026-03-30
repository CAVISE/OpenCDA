"""Helpers for resolving metric collector configuration from module configs."""

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def resolve_metric_collector_config(
    module_config: Mapping[str, Any] | None,
    default_metric_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """
    Resolve metric collector configuration from a module config.

    The resolved value is a single mapping from metric name to metric
    constructor parameters. The keys of this mapping define the enabled
    metrics.
    """
    metrics_config = dict((module_config or {}).get("metrics", {}))
    base_metric_configs = {metric_name: dict(params) for metric_name, params in (default_metric_configs or {}).items()}

    legacy_keys = [key for key in ("enabled_metrics", "metric_params") if key in metrics_config]
    if legacy_keys:
        logger.error(
            "Legacy metric config keys are not supported anymore: %s",
            ", ".join(sorted(legacy_keys)),
        )
        raise ValueError("Legacy metric config keys are not supported anymore: " + ", ".join(sorted(legacy_keys)))

    explicit_metric_configs = {metric_name: dict(params) for metric_name, params in metrics_config.get("metric_configs", {}).items()}

    if explicit_metric_configs:
        resolved_metric_configs = {}
        for metric_name, params in explicit_metric_configs.items():
            merged_params = dict(base_metric_configs.get(metric_name, {}))
            merged_params.update(params)
            resolved_metric_configs[metric_name] = merged_params
    elif base_metric_configs:
        resolved_metric_configs = base_metric_configs
    else:
        resolved_metric_configs = None

    logger.info(
        "Resolved metric config metric_configs=%s",
        None if resolved_metric_configs is None else sorted(resolved_metric_configs),
    )
    return resolved_metric_configs
