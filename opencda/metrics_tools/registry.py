"""Registry for discoverable metric classes."""

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Registry for metric classes.

    Metrics are keyed by `metric_name` and registered
    automatically from `BaseMetric.__init_subclass__`.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, metric_cls: type) -> None:
        """Register a concrete metric class."""
        if inspect.isabstract(metric_cls):
            raise ValueError(f"Cannot register abstract metric class '{metric_cls.__name__}'.")

        if (metric_name := getattr(metric_cls, "metric_name")) is None:
            raise ValueError(f"Metric class '{metric_cls.__name__}' must define 'metric_name'.")

        if metric_name in cls._registry:
            raise ValueError(f"Duplicate metric registration for metric='{metric_name}'.")

        cls._registry[metric_name] = metric_cls
        logger.debug("Registered metric class '%s' as '%s'.", metric_cls.__name__, metric_name)

    @classmethod
    def get_metric_class(cls, metric_name: str) -> type:
        """Return a metric class for the given metric name."""
        if metric_name not in cls._registry:
            available = cls.list_metrics()
            raise KeyError(f"Unknown metric '{metric_name}'. Available: {available}")
        return cls._registry[metric_name]

    @classmethod
    def create_metric(cls, metric_name: str, **kwargs: Any) -> Any:
        """Instantiate a metric by name."""
        metric_cls = cls.get_metric_class(metric_name=metric_name)
        return metric_cls(**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List registered metrics."""
        return list(cls._registry)
