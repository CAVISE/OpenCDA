"""Unit tests for opencda.metrics_tools.

Covers config, metric_collector, individual metrics (speed, acceleration,
distance_gap, time_gap, ttc, trace), report_builder, metric_sample,
and collection_models.
"""

from __future__ import annotations

import math

import pytest

from opencda.metrics_tools.config import resolve_metric_collector_config
from opencda.metrics_tools.metric_collector import MetricCollector
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.collection_models import (
    MetricCollection,
    MetricIssue,
    MetricSeries,
)
from opencda.metrics_tools.report_builder import UniversalReportBuilder
from opencda.metrics_tools.report_models import (
    EntityMetricCollections,
    MetricReportSpec,
    MetricSummarySpec,
)
from opencda.metrics_tools.metrics.speed import SpeedMetric
from opencda.metrics_tools.metrics.acceleration import AccelerationMetric
from opencda.metrics_tools.metrics.distance_gap import DistanceGapMetric
from opencda.metrics_tools.metrics.time_gap import TimeGapMetric
from opencda.metrics_tools.metrics.ttc import TtcMetric
from opencda.metrics_tools.metrics.localization_trace import LocalizationTraceMetric


# ---------------------------------------------------------------------------
# M-CFG: config.py
# ---------------------------------------------------------------------------


class TestResolveMetricCollectorConfig:
    """Tests for resolve_metric_collector_config."""

    def test_m_cfg_01_explicit_overrides_default(self):
        """Explicit metric_configs override default values."""
        module_config = {
            "metrics": {
                "metric_configs": {
                    "speed": {"warmup_steps": 50},
                },
            },
        }
        defaults = {
            "speed": {"warmup_steps": 100, "dt": 0.05},
        }
        result = resolve_metric_collector_config(module_config, defaults)
        assert result is not None
        assert "speed" in result
        assert result["speed"]["warmup_steps"] == 50
        assert result["speed"]["dt"] == 0.05

    def test_m_cfg_02_legacy_key_enabled_metrics_raises(self):
        """Legacy key 'enabled_metrics' raises ValueError."""
        module_config = {"metrics": {"enabled_metrics": ["speed"]}}
        with pytest.raises(ValueError, match="enabled_metrics"):
            resolve_metric_collector_config(module_config)

    def test_m_cfg_03_legacy_key_metric_params_raises(self):
        """Legacy key 'metric_params' raises ValueError."""
        module_config = {"metrics": {"metric_params": {"speed": {}}}}
        with pytest.raises(ValueError, match="metric_params"):
            resolve_metric_collector_config(module_config)

    def test_m_cfg_04_defaults_used_when_no_explicit(self):
        """When metric_configs is None but defaults exist, defaults are returned."""
        module_config = {}
        defaults = {
            "speed": {"warmup_steps": 100},
            "ttc": {"warmup_steps": 50},
        }
        result = resolve_metric_collector_config(module_config, defaults)
        assert result is not None
        assert set(result.keys()) == {"speed", "ttc"}
        assert result["speed"]["warmup_steps"] == 100
        assert result["ttc"]["warmup_steps"] == 50

    def test_m_cfg_05_both_empty_returns_none(self):
        """When both dicts are empty or None, returns None."""
        assert resolve_metric_collector_config(None, None) is None
        assert resolve_metric_collector_config({}, None) is None
        assert resolve_metric_collector_config(None, {}) is None
        assert resolve_metric_collector_config({"metrics": {}}, {}) is None


# ---------------------------------------------------------------------------
# M-CLC: metric_collector.py
# ---------------------------------------------------------------------------


class TestMetricCollector:
    """Tests for MetricCollector."""

    def test_m_clc_01_all_metrics_when_config_none(self):
        """When metric_configs is None, all registered metrics are active."""
        collector = MetricCollector(module="test", entity_id="e1", metric_configs=None)
        from opencda.metrics_tools.registry import MetricRegistry

        all_registered = set(MetricRegistry.list_metrics())
        assert set(collector.active_metrics) == all_registered

    def test_m_clc_02_specific_metrics_by_config(self):
        """Only specified metrics are activated when config is provided."""
        collector = MetricCollector(
            module="test",
            entity_id="e1",
            metric_configs={"speed": {}, "ttc": {}},
        )
        assert set(collector.active_metrics) == {"speed", "ttc"}

    def test_m_clc_03_unknown_metric_unsupported(self):
        """Unknown metric name goes into unsupported_metrics."""
        collector = MetricCollector(
            module="test",
            entity_id="e1",
            metric_configs={"speed": {}, "nonexistent_metric": {}},
        )
        assert "speed" in collector.active_metrics
        assert "nonexistent_metric" in collector.unsupported_metrics

    def test_m_clc_04_update_delegates_to_all_active(self):
        """update() delegates to all active metrics."""
        collector = MetricCollector(
            module="test",
            entity_id="e1",
            metric_configs={"speed": {"warmup_steps": 0}},
        )
        metric = collector.get_metric("speed")
        assert metric is not None

        collector.update({"ego_speed": 100.0})
        collector.update({"ego_speed": 72.0})

        speed_metric: SpeedMetric = metric
        assert len(speed_metric._speed_samples) == 2

    def test_m_clc_05_get_metric_returns_none_for_inactive(self):
        """get_metric returns None for a metric that was not configured."""
        collector = MetricCollector(
            module="test",
            entity_id="e1",
            metric_configs={"speed": {}},
        )
        assert collector.get_metric("ttc") is None

    def test_m_clc_06_get_raw_forms_metric_collection(self):
        """get_raw() returns MetricCollection with correct fields."""
        collector = MetricCollector(
            module="test_module",
            entity_id="e2",
            metric_configs={"speed": {"warmup_steps": 0}, "nonexistent": {}},
        )
        collector.update({"ego_speed": 50.0})

        raw = collector.get_raw()
        assert isinstance(raw, MetricCollection)
        assert raw.module == "test_module"
        assert raw.entity_id == "e2"
        assert "speed" in raw.active_metrics
        assert len(raw.series) > 0

        # unsupported_metrics should be MetricIssue tuples
        assert len(raw.unsupported_metrics) == 1
        issue = raw.unsupported_metrics[0]
        assert isinstance(issue, MetricIssue)
        assert issue.metric_name == "nonexistent"


# ---------------------------------------------------------------------------
# M-SPD: speed.py
# ---------------------------------------------------------------------------


class TestSpeedMetric:
    """Tests for SpeedMetric."""

    def test_m_spd_01_kmh_to_ms(self):
        """100 km/h is converted to ~27.778 m/s."""
        metric = SpeedMetric(warmup_steps=0)
        metric.update({"ego_speed": 100.0})
        assert len(metric._speed_samples) == 1
        assert math.isclose(metric._speed_samples[0].value, 100.0 / 3.6, rel_tol=1e-6)

    def test_m_spd_02_zero_speed(self):
        """Zero speed yields 0.0."""
        metric = SpeedMetric(warmup_steps=0)
        metric.update({"ego_speed": 0.0})
        assert metric._speed_samples[0].value == 0.0

    def test_m_spd_03_get_raw_returns_metric_series(self):
        """get_raw() returns one MetricSeries named 'speed'."""
        metric = SpeedMetric(warmup_steps=0)
        metric.update({"ego_speed": 36.0})
        raw = metric.get_raw()
        assert len(raw) == 1
        assert raw[0].name == "speed"


# ---------------------------------------------------------------------------
# M-ACC: acceleration.py
# ---------------------------------------------------------------------------


class TestAccelerationMetric:
    """Tests for AccelerationMetric."""

    def test_m_acc_01_calculation(self):
        """Acceleration = (v2 - v1) / dt for 72->108 km/h, dt=0.05."""
        metric = AccelerationMetric(warmup_steps=0, dt=0.05)
        metric.update({"ego_speed": 72.0})
        metric.update({"ego_speed": 108.0})

        assert len(metric._acceleration_samples) == 2
        # First step: no previous speed -> 0.0
        assert metric._acceleration_samples[0].value == 0.0
        # Second step: (108/3.6 - 72/3.6) / 0.05 = (30.0 - 20.0) / 0.05 = 200
        assert math.isclose(metric._acceleration_samples[1].value, 200.0, rel_tol=1e-6)

    def test_m_acc_02_first_step_zero(self):
        """First step has no previous speed, so acceleration is 0.0."""
        metric = AccelerationMetric(warmup_steps=0, dt=0.05)
        metric.update({"ego_speed": 50.0})
        assert metric._acceleration_samples[0].value == 0.0

    def test_m_acc_03_custom_dt(self):
        """Custom dt is used for acceleration calculation."""
        metric = AccelerationMetric(warmup_steps=0, dt=0.1)
        metric.update({"ego_speed": 0.0})
        metric.update({"ego_speed": 36.0})

        # (10.0 - 0.0) / 0.1 = 100.0
        assert math.isclose(metric._acceleration_samples[1].value, 100.0, rel_tol=1e-6)

    def test_m_acc_04_no_ego_speed(self):
        """When ego_speed is missing from context, 0.0 is used."""
        metric = AccelerationMetric(warmup_steps=0, dt=0.05)
        metric.update({})
        assert metric._acceleration_samples[0].value == 0.0


# ---------------------------------------------------------------------------
# M-DG: distance_gap.py
# ---------------------------------------------------------------------------


class TestDistanceGapMetric:
    """Tests for DistanceGapMetric."""

    def test_m_dg_01_reads_from_context(self):
        """distance_gap value from context is stored."""
        metric = DistanceGapMetric(warmup_steps=0)
        metric.update({"distance_gap": 42.5})
        assert metric._samples[0].value == 42.5

    def test_m_dg_02_default_when_absent(self):
        """Default value is 100.0 when distance_gap is absent."""
        metric = DistanceGapMetric(warmup_steps=0)
        metric.update({})
        assert metric._samples[0].value == 100.0

    def test_m_dg_03_cutoff_in_summary(self):
        """Values >= cutoff are filtered from summary via report_builder."""
        metric = DistanceGapMetric(warmup_steps=0)
        for val in [50.0, 99.9, 100.0, 150.0, 30.0]:
            metric.update({"distance_gap": val})

        raw = metric.get_raw()
        spec = metric.get_report_spec()
        builder = UniversalReportBuilder()
        summaries = builder.build_metric_summaries(
            spec,
            lambda series_name: [s.value for s in raw[0].samples] if raw[0].name == series_name else [],
        )
        assert len(summaries) == 1
        summary = summaries[0]
        # cutoff is 100.0, strict <: only 50.0, 99.9, 30.0
        assert summary.count == 3


# ---------------------------------------------------------------------------
# M-TG: time_gap.py
# ---------------------------------------------------------------------------


class TestTimeGapMetric:
    """Tests for TimeGapMetric."""

    def test_m_tg_01_reads_from_context(self):
        """time_gap value from context is stored."""
        metric = TimeGapMetric(warmup_steps=0)
        metric.update({"time_gap": 3.5})
        assert metric._samples[0].value == 3.5

    def test_m_tg_02_default_when_absent(self):
        """Default value is 100.0 when time_gap is absent."""
        metric = TimeGapMetric(warmup_steps=0)
        metric.update({})
        assert metric._samples[0].value == 100.0

    def test_m_tg_03_cutoff_in_summary(self):
        """Values >= cutoff are filtered from summary."""
        metric = TimeGapMetric(warmup_steps=0)
        for val in [50.0, 99.9, 100.0, 150.0]:
            metric.update({"time_gap": val})

        raw = metric.get_raw()
        spec = metric.get_report_spec()
        builder = UniversalReportBuilder()
        summaries = builder.build_metric_summaries(
            spec,
            lambda series_name: [s.value for s in raw[0].samples] if raw[0].name == series_name else [],
        )
        assert summaries[0].count == 2  # 50.0 and 99.9


# ---------------------------------------------------------------------------
# M-TTC: ttc.py
# ---------------------------------------------------------------------------


class TestTtcMetric:
    """Tests for TtcMetric."""

    def test_m_ttc_01_reads_from_context(self):
        """ttc value from context is stored."""
        metric = TtcMetric(warmup_steps=0)
        metric.update({"ttc": 5.0})
        assert metric._ttc_samples[0].value == 5.0

    def test_m_ttc_02_default_when_absent(self):
        """Default value is 1000.0 when ttc is absent."""
        metric = TtcMetric(warmup_steps=0)
        metric.update({})
        assert metric._ttc_samples[0].value == 1000.0

    def test_m_ttc_03_valid_statistics_normal(self):
        """valid_statistics returns (mean, std) for normal data."""
        metric = TtcMetric(warmup_steps=0)
        for val in [500.0, 600.0, 700.0]:
            metric.update({"ttc": val})

        mean, std = metric.valid_statistics(cutoff=1000.0)
        assert mean is not None
        assert std is not None
        assert math.isclose(mean, 600.0, rel_tol=1e-6)

    def test_m_ttc_04_valid_statistics_all_above_cutoff(self):
        """valid_statistics returns (None, None) when all values >= cutoff."""
        metric = TtcMetric(warmup_steps=0)
        for val in [1000.0, 1500.0, 2000.0]:
            metric.update({"ttc": val})

        mean, std = metric.valid_statistics(cutoff=1000.0)
        assert mean is None
        assert std is None


# ---------------------------------------------------------------------------
# M-TR: localization_trace.py
# ---------------------------------------------------------------------------


class TestLocalizationTraceMetric:
    """Tests for LocalizationTraceMetric."""

    def test_m_tr_01_all_12_series_created(self):
        """get_raw() returns 12 MetricSeries."""
        metric = LocalizationTraceMetric(warmup_steps=0)
        metric.update({name: 0.0 for name in LocalizationTraceMetric._SERIES_NAMES})
        raw = metric.get_raw()
        assert len(raw) == 12

    def test_m_tr_02_speed_series_converted(self):
        """Only speed series (gnss_speed, filter_speed, gt_speed) are divided by 3.6."""
        metric = LocalizationTraceMetric(warmup_steps=0)
        metric.update(
            {
                "gnss_x": 100.0,
                "gnss_y": 200.0,
                "gnss_yaw": 1.5,
                "gnss_speed": 36.0,
                "filter_x": 101.0,
                "filter_y": 201.0,
                "filter_yaw": 1.6,
                "filter_speed": 72.0,
                "gt_x": 102.0,
                "gt_y": 202.0,
                "gt_yaw": 1.7,
                "gt_speed": 108.0,
            }
        )
        raw = metric.get_raw()
        series_by_name = {s.name: s for s in raw}

        # Speed series should be divided by 3.6
        assert math.isclose(series_by_name["gnss_speed"].samples[0].value, 10.0, rel_tol=1e-6)
        assert math.isclose(series_by_name["filter_speed"].samples[0].value, 20.0, rel_tol=1e-6)
        assert math.isclose(series_by_name["gt_speed"].samples[0].value, 30.0, rel_tol=1e-6)

        # Non-speed series should NOT be divided
        assert series_by_name["gnss_x"].samples[0].value == 100.0
        assert series_by_name["gnss_y"].samples[0].value == 200.0
        assert series_by_name["gnss_yaw"].samples[0].value == 1.5

    def test_m_tr_03_custom_resolver_absolute_error(self):
        """Resolver returns absolute error between gt_x and gnss_x."""
        metric = LocalizationTraceMetric(warmup_steps=0)
        metric.update(
            {
                "gnss_x": 100.0,
                "gt_x": 102.0,
                "gnss_speed": 0.0,
                "filter_x": 0.0,
                "filter_y": 0.0,
                "filter_yaw": 0.0,
                "filter_speed": 0.0,
                "gnss_y": 0.0,
                "gnss_yaw": 0.0,
                "gt_y": 0.0,
                "gt_yaw": 0.0,
                "gt_speed": 0.0,
            }
        )

        spec = metric.get_report_spec()
        resolver_spec = spec.summary_specs[0]  # GNSS raw data x-axis error
        assert resolver_spec.resolver is not None

        # Build a value resolver that returns values from the metric
        series_map = {s.name: [sample.value for sample in s.samples] for s in metric.get_raw()}
        values = resolver_spec.resolver(lambda name: series_map.get(name, []))
        assert len(values) == 1
        assert math.isclose(values[0], 2.0, rel_tol=1e-6)

    def test_m_tr_04_absolute_difference_empty(self):
        """_absolute_difference returns () for empty inputs."""
        assert LocalizationTraceMetric._absolute_difference([], []) == ()
        assert LocalizationTraceMetric._absolute_difference([], [1.0]) == ()
        assert LocalizationTraceMetric._absolute_difference([1.0], []) == ()

    def test_m_tr_05_signed_difference(self):
        """_signed_difference returns (a - b) for each pair."""
        result = LocalizationTraceMetric._signed_difference([5.0, 3.0], [3.0, 5.0])
        assert len(result) == 2
        assert math.isclose(result[0], 2.0, rel_tol=1e-6)
        assert math.isclose(result[1], -2.0, rel_tol=1e-6)

    def test_m_tr_06_absolute_difference_different_lengths(self):
        """_absolute_difference zips, so result length = min(len1, len2)."""
        result = LocalizationTraceMetric._absolute_difference([1.0, 2.0, 3.0], [10.0, 20.0])
        assert len(result) == 2
        assert math.isclose(result[0], 9.0, rel_tol=1e-6)
        assert math.isclose(result[1], 18.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# M-RPT: report_builder.py
# ---------------------------------------------------------------------------


class TestReportBuilder:
    """Tests for UniversalReportBuilder."""

    def test_m_rpt_01_build_summary_normal(self):
        """build_summary computes correct count, mean, std, min, max."""
        builder = UniversalReportBuilder()
        summary = builder.build_summary("test", [10.0, 20.0, 30.0])
        assert summary.count == 3
        assert math.isclose(summary.mean, 20.0, rel_tol=1e-6)
        assert summary.std is not None
        assert summary.std > 0
        assert summary.min == 10.0
        assert summary.max == 30.0

    def test_m_rpt_02_build_summary_cutoff(self):
        """build_summary filters values >= cutoff."""
        builder = UniversalReportBuilder()
        summary = builder.build_summary("test", [5.0, 10.0, 15.0, 20.0], cutoff=15.0)
        assert summary.count == 2
        assert summary.max == 10.0

    def test_m_rpt_03_build_summary_empty(self):
        """build_summary with empty data returns None for mean/std/min/max."""
        builder = UniversalReportBuilder()
        summary = builder.build_summary("test", [])
        assert summary.count == 0
        assert summary.mean is None
        assert summary.std is None
        assert summary.min is None
        assert summary.max is None

    def test_m_rpt_04_all_filtered_by_cutoff(self):
        """When all values are filtered, returns count=0 and None stats."""
        builder = UniversalReportBuilder()
        summary = builder.build_summary("test", [10.0, 20.0], cutoff=5.0)
        assert summary.count == 0
        assert summary.mean is None
        assert summary.std is None

    def test_m_rpt_05_build_metric_summaries_multiple_specs(self):
        """build_metric_summaries processes each spec independently."""
        builder = UniversalReportBuilder()
        spec = MetricReportSpec(
            metric_name="test",
            summary_specs=(
                MetricSummarySpec(series_name="a"),
                MetricSummarySpec(series_name="b", cutoff=50.0),
            ),
        )

        def value_resolver(name):
            return {"a": [10.0, 20.0], "b": [30.0, 60.0]}[name]

        summaries = builder.build_metric_summaries(spec, value_resolver)
        assert len(summaries) == 2
        assert summaries[0].count == 2  # a: both included
        assert summaries[1].count == 1  # b: 60.0 >= 50.0 filtered

    def test_m_rpt_06_resolver_priority_over_series_name(self):
        """_resolve_summary_values uses resolver when present, ignoring series_name."""
        builder = UniversalReportBuilder()
        spec = MetricSummarySpec(
            series_name="unused",
            resolver=lambda vr: [42.0, 84.0],
        )

        def value_resolver(name):
            return [1.0, 2.0]

        result = builder._resolve_summary_values(spec, value_resolver)
        assert list(result) == [42.0, 84.0]

    def test_m_rpt_07_build_entity_report_meta(self):
        """build_entity_report preserves meta info from MetricCollection."""
        builder = UniversalReportBuilder()
        collection = MetricCollection(
            module="mod1",
            entity_id="ent1",
            active_metrics=("speed",),
            disabled_metrics=("ttc",),
            unsupported_metrics=(MetricIssue(metric_name="foo", reason="bar"),),
            series=(
                MetricSeries(
                    name="speed",
                    samples=(MetricSample(tick=1, value=10.0),),
                ),
            ),
        )
        report = builder.build_entity_report(collection)
        assert report.info.module == "mod1"
        assert report.info.entity_id == "ent1"
        assert report.info.active_metrics == ("speed",)
        assert report.info.disabled_metrics == ("ttc",)
        assert len(report.info.unsupported_metrics) == 1
        assert report.info.unsupported_metrics[0].metric_name == "foo"

    def test_m_rpt_08_build_module_and_group_reports(self):
        """build_module_report and build_group_report produce correct structures."""
        builder = UniversalReportBuilder()

        entity_collection = MetricCollection(
            module="mod1",
            entity_id="ent1",
            active_metrics=("speed",),
            disabled_metrics=(),
            unsupported_metrics=(),
            series=(
                MetricSeries(
                    name="speed",
                    samples=(MetricSample(tick=1, value=10.0),),
                ),
            ),
        )
        entity_report = builder.build_entity_report(entity_collection)

        module_report = builder.build_module_report("mod1", (entity_report,))
        assert module_report.module == "mod1"
        assert len(module_report.entities) == 1

        entity_collections = EntityMetricCollections(
            entity_id="ent1",
            context_id="ctx1",
            collections=(entity_collection,),
        )
        group_report = builder.build_group_report("g1", (entity_collections,), "mod1")
        assert group_report.group_id == "g1"
        assert len(group_report.entities) == 1

    def test_m_rpt_09_cutoff_strict_inequality(self):
        """Cutoff uses strict inequality (<); value == cutoff is excluded."""
        builder = UniversalReportBuilder()
        summary = builder.build_summary("test", [99.0, 100.0, 101.0], cutoff=100.0)
        assert summary.count == 1
        assert summary.min == 99.0

    def test_m_rpt_10_to_dict(self):
        """to_dict() on EntityReport/ModuleReport/GroupReport works without errors."""
        builder = UniversalReportBuilder()
        collection = MetricCollection(
            module="mod1",
            entity_id="ent1",
            active_metrics=("speed",),
            disabled_metrics=(),
            unsupported_metrics=(),
            series=(
                MetricSeries(
                    name="speed",
                    samples=(MetricSample(tick=1, value=10.0),),
                ),
            ),
        )
        entity_report = builder.build_entity_report(collection)
        entity_dict = entity_report.to_dict()
        assert isinstance(entity_dict, dict)
        assert entity_dict["info"]["module"] == "mod1"

        module_report = builder.build_module_report("mod1", (entity_report,))
        module_dict = module_report.to_dict()
        assert isinstance(module_dict, dict)

        entity_collections = EntityMetricCollections(
            entity_id="ent1",
            collections=(collection,),
        )
        group_report = builder.build_group_report("g1", (entity_collections,), "mod1")
        group_dict = group_report.to_dict()
        assert isinstance(group_dict, dict)


# ---------------------------------------------------------------------------
# M-SAMPLE: metric_sample.py
# ---------------------------------------------------------------------------


class TestMetricSample:
    """Tests for MetricSample."""

    def test_m_sample_01_to_dict(self):
        """to_dict() returns {'tick': tick, 'value': value}."""
        sample = MetricSample(tick=5, value=42.0)
        d = sample.to_dict()
        assert d == {"tick": 5, "value": 42.0}


# ---------------------------------------------------------------------------
# M-COLLECT: collection_models.py
# ---------------------------------------------------------------------------


class TestCollectionModels:
    """Tests for MetricCollection."""

    def test_m_collect_01_get_series(self):
        """get_series returns samples for existing series and empty tuple for missing."""
        sample1 = MetricSample(tick=1, value=10.0)
        sample2 = MetricSample(tick=2, value=20.0)
        collection = MetricCollection(
            module="m",
            entity_id="e",
            active_metrics=(),
            disabled_metrics=(),
            unsupported_metrics=(),
            series=(MetricSeries(name="speed", samples=(sample1, sample2)),),
        )
        assert len(collection.get_series("speed")) == 2
        assert collection.get_series("nonexistent") == ()

    def test_m_collect_02_to_dict(self):
        """MetricCollection.to_dict() serializes all fields."""
        collection = MetricCollection(
            module="m",
            entity_id="e",
            active_metrics=("speed",),
            disabled_metrics=("ttc",),
            unsupported_metrics=(MetricIssue(metric_name="foo", reason="bar"),),
            series=(MetricSeries(name="speed", samples=(MetricSample(tick=1, value=10.0),)),),
        )
        d = collection.to_dict()
        assert isinstance(d, dict)
        assert d["module"] == "m"
        assert d["entity_id"] == "e"
        assert len(d["active_metrics"]) == 1
        assert len(d["series"]) == 1
