from typing import Any, Callable, Mapping, Sequence

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class LocalizationTraceMetric(BaseMetric):
    """Collect raw localization traces for report generation."""

    metric_name = "trace"

    _SERIES_NAMES = (
        "gnss_x",
        "gnss_y",
        "gnss_yaw",
        "gnss_speed",
        "filter_x",
        "filter_y",
        "filter_yaw",
        "filter_speed",
        "gt_x",
        "gt_y",
        "gt_yaw",
        "gt_speed",
    )
    _SPEED_SERIES = {"gnss_speed", "filter_speed", "gt_speed"}

    @classmethod
    def supports(cls, capabilities: Mapping[str, Any] | None = None) -> bool:
        if not capabilities:
            return False
        return bool(capabilities.get("gnss") and capabilities.get("filter") and capabilities.get("ground_truth"))

    def __init__(self, warmup_steps: int = 0):
        super().__init__(warmup_steps=warmup_steps)
        self._samples: dict[str, list[MetricSample]] = {series_name: [] for series_name in self._SERIES_NAMES}

    def _process_context(self, context: Mapping[str, object]) -> None:
        for series_name in self._SERIES_NAMES:
            value = float(context.get(series_name, 0.0))
            if series_name in self._SPEED_SERIES:
                value /= 3.6
            self._samples[series_name].append(self._make_sample(value))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(
            MetricSeries(name=series_name, samples=tuple(self._samples[series_name]))
            for series_name in self._SERIES_NAMES
        )

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Localization Trace",
            series_names=cls._SERIES_NAMES,
            summary_specs=(
                MetricSummarySpec(
                    display_name="GNSS raw data x-axis error",
                    resolver=cls._absolute_error_resolver("gt_x", "gnss_x"),
                ),
                MetricSummarySpec(
                    display_name="GNSS raw data y-axis error",
                    resolver=cls._absolute_error_resolver("gt_y", "gnss_y"),
                ),
                MetricSummarySpec(
                    display_name="GNSS raw data yaw error",
                    resolver=cls._absolute_error_resolver("gt_yaw", "gnss_yaw"),
                ),
                MetricSummarySpec(
                    display_name="Data fusion x-axis error",
                    resolver=cls._absolute_error_resolver("gt_x", "filter_x"),
                ),
                MetricSummarySpec(
                    display_name="Data fusion y-axis error",
                    resolver=cls._absolute_error_resolver("gt_y", "filter_y"),
                ),
                MetricSummarySpec(
                    display_name="Data fusion yaw error",
                    resolver=cls._absolute_error_resolver("gt_yaw", "filter_yaw"),
                ),
            ),
        )

    @staticmethod
    def _absolute_error_resolver(
        base_series: str,
        compared_series: str,
    ) -> Callable[[Callable[[str], Sequence[float]]], Sequence[float]]:
        return lambda value_resolver: LocalizationTraceMetric._absolute_difference(
            value_resolver(base_series),
            value_resolver(compared_series),
        )

    @staticmethod
    def _absolute_difference(
        first_values: Sequence[float],
        second_values: Sequence[float],
    ) -> Sequence[float]:
        if not first_values or not second_values:
            return ()
        return tuple(float(abs(a - b)) for a, b in zip(first_values, second_values))

    @staticmethod
    def _signed_difference(
        first_values: Sequence[float],
        second_values: Sequence[float],
    ) -> Sequence[float]:
        if not first_values or not second_values:
            return ()
        return tuple(float(a - b) for a, b in zip(first_values, second_values))
