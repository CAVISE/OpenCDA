"""Universal report building primitives for metric-based modules."""

import logging
from typing import Callable, Sequence
import numpy as np

from opencda.metrics_tools.collection_models import MetricCollection
from opencda.metrics_tools.registry import MetricRegistry
from opencda.metrics_tools.report_models import (
    EntityMetricCollections,
    EntityReport,
    EntityReportInfo,
    GroupReport,
    MetricReport,
    MetricReportSpec,
    MetricSummarySpec,
    ModuleReport,
    SeriesSummary,
)
from opencda.metrics_tools.base_metric import BaseMetric

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.report_builder")


class UniversalReportBuilder:
    """Universal summary builder over resolved metric series."""

    def build_metric_summaries(
        self,
        metric_spec: MetricReportSpec,
        value_resolver: Callable[[str], Sequence[float]],
    ) -> tuple[SeriesSummary, ...]:
        return tuple(
            self.build_summary(
                spec.display_name or spec.series_name or "",
                self._resolve_summary_values(spec, value_resolver),
                cutoff=spec.cutoff,
            )
            for spec in metric_spec.summary_specs
        )

    def build_summary(
        self,
        name: str,
        values: Sequence[float],
        cutoff: float | None = None,
    ) -> SeriesSummary:
        series_values = np.array(list(values), dtype=float)
        if cutoff is not None:
            series_values = series_values[series_values < cutoff]

        if len(series_values) == 0:
            return SeriesSummary(name=name, count=0, mean=None, std=None, min=None, max=None)

        return SeriesSummary(
            name=name,
            count=int(len(series_values)),
            mean=float(np.mean(series_values)),
            std=float(np.std(series_values)),
            min=float(np.min(series_values)),
            max=float(np.max(series_values)),
        )

    @staticmethod
    def _resolve_summary_values(
        spec: MetricSummarySpec,
        value_resolver: Callable[[str], Sequence[float]],
    ) -> Sequence[float]:
        if spec.resolver is not None:
            return spec.resolver(value_resolver)
        if spec.series_name is None:
            return ()
        return value_resolver(spec.series_name)

    def build_entity_report(self, raw_data: MetricCollection) -> EntityReport:
        logger.info(
            "Building entity report module=%s entity_id=%s active_metrics=%s",
            raw_data.module,
            raw_data.entity_id,
            raw_data.active_metrics,
        )
        return EntityReport(
            info=EntityReportInfo(
                module=raw_data.module,
                entity_id=raw_data.entity_id,
                active_metrics=raw_data.active_metrics,
                disabled_metrics=raw_data.disabled_metrics,
                unsupported_metrics=raw_data.unsupported_metrics,
            ),
            metrics=tuple(self._build_metric_report(metric_cls, raw_data) for metric_cls in _active_metric_classes(raw_data.active_metrics)),
        )

    def build_module_report(self, module: str, entities: Sequence[EntityReport]) -> ModuleReport:
        logger.info("Building module report module=%s entities=%d", module, len(entities))
        return ModuleReport(module=module, entities=tuple(entities))

    def build_group_report(
        self,
        group_id: int | str,
        entities: Sequence[EntityMetricCollections],
        module: str,
    ) -> GroupReport:
        logger.info(
            "Building group report group_id=%s module=%s entities=%d",
            group_id,
            module,
            len(entities),
        )
        return GroupReport(
            group_id=group_id,
            entities=tuple(self._build_group_entity_report(entity, module) for entity in entities),
        )

    def _build_group_entity_report(
        self,
        entity_collections: EntityMetricCollections,
        module: str,
    ) -> EntityReport:
        collection = entity_collections.get_collection(module)
        if collection is None:
            return EntityReport(
                info=EntityReportInfo(
                    module=module,
                    entity_id=entity_collections.entity_id,
                    context_id=entity_collections.context_id,
                )
            )

        entity_report = self.build_entity_report(collection)
        return EntityReport(
            info=EntityReportInfo(
                module=entity_report.info.module,
                entity_id=entity_report.info.entity_id,
                context_id=entity_collections.context_id,
                active_metrics=entity_report.info.active_metrics,
                disabled_metrics=entity_report.info.disabled_metrics,
                unsupported_metrics=entity_report.info.unsupported_metrics,
            ),
            metrics=entity_report.metrics,
        )

    def _build_metric_report(self, metric_cls: type[BaseMetric], raw_data: MetricCollection) -> MetricReport:
        metric_spec = metric_cls.get_report_spec()
        return MetricReport(
            metric_name=metric_spec.metric_name,
            display_name=metric_spec.display_name or metric_spec.metric_name,
            summary=self.build_metric_summaries(metric_spec, lambda series_name: _collection_values(raw_data, series_name)),
            series=_metric_series(raw_data, metric_spec.series_names),
        )


def _collection_values(metric_collection: MetricCollection, series_name: str) -> list[float]:
    return [sample.value for sample in metric_collection.get_series(series_name)]


def _active_metric_classes(active_metrics: tuple[str, ...]) -> list[type]:
    return [MetricRegistry.get_metric_class(metric_name=metric_name) for metric_name in active_metrics]


def _metric_series(metric_collection: MetricCollection, series_names: tuple[str, ...]) -> tuple:
    series_name_set = set(series_names)
    return tuple(series for series in metric_collection.series if series.name in series_name_set)
