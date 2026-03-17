import inspect
from typing import Any


class MetricRegistry:
    """
    Registry for metric classes.

    Metrics are keyed by `(module, metric_name)` and registered
    automatically from `BaseMetric.__init_subclass__`.
    """

    _registry: dict[tuple[str, str], type] = {}

    @classmethod
    def register(cls, metric_cls: type) -> None:
        """Register a concrete metric class."""
        if inspect.isabstract(metric_cls):
            return

        module = getattr(metric_cls, "module", None)
        metric_name = getattr(metric_cls, "metric_name", None)
        if not module or not metric_name:
            raise ValueError(f"Metric class '{metric_cls.__name__}' must define both 'module' and 'metric_name'.")

        registry_key = (module, metric_name)
        if registry_key in cls._registry:
            raise ValueError(f"Duplicate metric registration for module='{module}', metric='{metric_name}'.")

        cls._registry[registry_key] = metric_cls

    @classmethod
    def get_metric_class(cls, module: str, metric_name: str) -> type:
        """Return a metric class for the given module and metric name."""
        registry_key = (module, metric_name)
        if registry_key not in cls._registry:
            available = cls.list_metrics(module=module)
            raise KeyError(f"Unknown metric '{metric_name}' for module '{module}'. Available: {available}")
        return cls._registry[registry_key]

    @classmethod
    def create_metric(cls, module: str, metric_name: str, **kwargs: Any) -> Any:
        """Instantiate a metric by name."""
        metric_cls = cls.get_metric_class(module=module, metric_name=metric_name)
        return metric_cls(**kwargs)

    @classmethod
    def list_metrics(cls, module: str | None = None) -> list[str]:
        """List registered metrics, optionally filtered by module."""
        if module is None:
            return [metric_name for _, metric_name in cls._registry]
        return [metric_name for registered_module, metric_name in cls._registry if registered_module == module]
