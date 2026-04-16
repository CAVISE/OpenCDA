"""
Metrics tools public API.

Provides:
- BaseMetric base class
- MetricCollector runtime collector
- get_metric_class lookup function
- create_metric factory function
- list_metrics utility
"""

import importlib

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricCollection, MetricIssue, MetricSeries
from opencda.metrics_tools.config import resolve_metric_collector_config
from opencda.metrics_tools.metric_collector import MetricCollector
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_builder import UniversalReportBuilder
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
from opencda.metrics_tools.registry import MetricRegistry

# Initialize builtin metric discovery.
importlib.import_module("opencda.metrics_tools.metrics")

get_metric_class = MetricRegistry.get_metric_class
create_metric = MetricRegistry.create_metric
list_metrics = MetricRegistry.list_metrics

__all__ = [
    "BaseMetric",
    "create_metric",
    "EntityMetricCollections",
    "EntityReport",
    "EntityReportInfo",
    "GroupReport",
    "get_metric_class",
    "MetricCollection",
    "MetricCollector",
    "MetricIssue",
    "MetricReport",
    "MetricReportSpec",
    "MetricSample",
    "MetricSeries",
    "MetricSummarySpec",
    "ModuleReport",
    "list_metrics",
    "resolve_metric_collector_config",
    "SeriesSummary",
    "UniversalReportBuilder",
]
