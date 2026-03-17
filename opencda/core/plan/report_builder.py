"""
Planning report builders operating on structured metric collection data.
"""

from abc import ABC, abstractmethod
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np

import opencda.core.plan.drive_profile_plotting as open_plt
from opencda.core.plan.metrics_tools.metric_sample import MetricSample
from opencda.core.plan.report_models import MetricCollection, PlanningActorReport, SeriesSummary


class ReportBuilder(ABC):
    """Abstract base class for report builders."""

    @abstractmethod
    def build(self, raw_data: MetricCollection) -> Any:
        """Build a report artifact from normalized raw metric data."""


class PlanningJsonReportBuilder(ReportBuilder):
    """Build a structured planning report from raw metric data."""

    TTC_CUTOFF = 1000.0

    def build(self, raw_data: MetricCollection) -> PlanningActorReport:
        speed_samples = self._get_samples(raw_data, "speed")
        acceleration_samples = self._get_samples(raw_data, "acceleration")
        ttc_samples = self._get_samples(raw_data, "ttc")

        return PlanningActorReport(
            module=raw_data.module,
            entity_id=raw_data.entity_id,
            active_metrics=raw_data.active_metrics,
            disabled_metrics=raw_data.disabled_metrics,
            unsupported_metrics=raw_data.unsupported_metrics,
            summary=(
                self._summary("speed", speed_samples),
                self._summary("acceleration", acceleration_samples),
                self._summary("ttc", ttc_samples, cutoff=self.TTC_CUTOFF),
            ),
            series=raw_data.series,
        )

    @staticmethod
    def _get_samples(raw_data: MetricCollection, series_name: str) -> tuple[MetricSample, ...]:
        return raw_data.get_series(series_name)

    @staticmethod
    def _summary(
        series_name: str,
        samples: tuple[MetricSample, ...],
        cutoff: float | None = None,
    ) -> SeriesSummary:
        values = np.array([sample.value for sample in samples], dtype=float)
        if cutoff is not None:
            values = values[values < cutoff]

        if len(values) == 0:
            return SeriesSummary(name=series_name, count=0, mean=None, std=None, min=None, max=None)

        return SeriesSummary(
            name=series_name,
            count=int(len(values)),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
        )


class PlanningPlotReportBuilder(ReportBuilder):
    """Build a planning matplotlib figure from raw metric data."""

    def build(self, raw_data: MetricCollection) -> plt.Figure:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            speed_values = [sample.value for sample in self._get_samples(raw_data, "speed")]
            acceleration_values = [sample.value for sample in self._get_samples(raw_data, "acceleration")]
            ttc_values = [sample.value for sample in self._get_samples(raw_data, "ttc")]

            figure = plt.figure()

            plt.subplot(311)
            open_plt.draw_velocity_profile_single_plot([speed_values])

            plt.subplot(312)
            open_plt.draw_acceleration_profile_single_plot([acceleration_values])

            plt.subplot(313)
            open_plt.draw_ttc_profile_single_plot([ttc_values])

            figure.suptitle(f"planning profile of actor id {raw_data.entity_id}")
            return figure

    @staticmethod
    def _get_samples(raw_data: MetricCollection, series_name: str) -> tuple[MetricSample, ...]:
        return raw_data.get_series(series_name)
