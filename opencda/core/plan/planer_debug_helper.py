"""
Runtime helper for planning metrics collection.
"""

from typing import Any, Mapping

from opencda.core.plan.metrics_tools.metric_collector import MetricCollector
from opencda.core.plan.metrics_tools.metrics.dynamics import DynamicsMetric
from opencda.core.plan.metrics_tools.metrics.ttc import TtcMetric
from opencda.core.plan.report_models import MetricCollection


class PlanDebugHelper(object):
    """
    Runtime wrapper around a planning metric collector.

    Parameters
    ----------
    actor_id : int
        The actor ID of the target vehicle.
    warmup_steps : int
        The number of steps to ignore at the beginning.
    """

    def __init__(
        self,
        actor_id: int,
        warmup_steps: int = 100,
        enabled_metrics: list[str] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ):
        self.actor_id = actor_id

        metric_params = {
            "dynamics": {"warmup_steps": warmup_steps},
            "ttc": {"warmup_steps": warmup_steps},
        }
        self.metric_collector = MetricCollector(
            module="planning",
            entity_id=actor_id,
            enabled_metrics=enabled_metrics,
            capabilities=capabilities,
            metric_params=metric_params,
        )

        self.dynamics_metric = self._require_metric("dynamics")
        self.ttc_metric = self._require_metric("ttc")

    @property
    def speed_list(self) -> list[float]:
        return self.dynamics_metric.speed_list

    @property
    def acc_list(self) -> list[float]:
        return self.dynamics_metric.acc_list

    @property
    def ttc_list(self) -> list[float]:
        return self.ttc_metric.ttc_list

    def update(self, *args: Any, context: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """
        Update all metrics with new data.

        Parameters
        ----------
        *args : Any
            Positional values for transitional call sites. Only
            `(ego_speed, ttc)` is accepted.
        context : Mapping[str, Any] | None
            Normalized runtime context.
        **kwargs : Any
            Additional context fields.
        """
        normalized_context = self._normalize_context(args, context, kwargs)
        self.metric_collector.update(normalized_context)

    def get_raw(self) -> MetricCollection:
        """Expose normalized raw data collected for planning metrics."""
        return self.metric_collector.get_raw()

    def _normalize_context(
        self,
        args: tuple[Any, ...],
        context: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if args and len(args) != 2:
            raise TypeError("PlanDebugHelper.update accepts either a context mapping or positional (ego_speed, ttc).")

        normalized: dict[str, Any] = {}
        if context is not None:
            normalized.update(dict(context))
        if args:
            normalized.update({"ego_speed": args[0], "ttc": args[1]})
        normalized.update(kwargs)
        return normalized

    def _require_metric(self, metric_name: str) -> Any:
        metric = self.metric_collector.get_metric(metric_name)
        if metric is None:
            raise ValueError(f"Required metric '{metric_name}' is not active for planning collector.")
        return metric
