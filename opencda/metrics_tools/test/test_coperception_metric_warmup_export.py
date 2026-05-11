from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from opencda.metrics_tools.metric_collector import MetricCollector
from opencda.metrics_tools.metrics.coperception import (
    _opencood_eval,
    ap_at_iou,
    attack_success_rate,
    attacker_benign_visibility_ratio,
    attacker_target_confidence,
    mean_precision_at_iou,
    mean_recall_at_iou,
)
from opencda.metrics_tools.report_builder import UniversalReportBuilder


class DummyEvalUtils:
    @staticmethod
    def caluclate_tp_fp(_pred_box_tensor, _pred_score, gt_box_tensor, result_stat, iou):
        result_stat[iou]["fp"].append(0)
        result_stat[iou]["tp"].append(1)
        result_stat[iou]["gt"] += gt_box_tensor.shape[0]
        result_stat[iou]["score"].append(1.0)

    @staticmethod
    def calculate_ap(_result_stat, _iou, _global_sort_detections):
        return 0.5, [0.0, 0.25, 0.75, 1.0], [0.0, 0.8, 0.6, 0.0]


class DummyPolygon:
    @property
    def centroid(self):
        return self

    def contains(self, _):
        return True


class DummyCommonUtils:
    @staticmethod
    def convert_format(boxes):
        return [DummyPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.ones(len(pred_polygon_list), dtype=np.float32)


class DummyAttackHelper:
    @staticmethod
    def resolve_ego_agent_id(_scenario_data):
        return "ego"

    @staticmethod
    def load_agent_state(_scenario_data, agent_id):
        return {
            "agent_id": agent_id,
            "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ego_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

    @staticmethod
    def resolve_agent_snapshot(scenario_data, agent_id):
        return scenario_data[agent_id]["000001"]

    @staticmethod
    def require_config_value(config, key):
        return config[key]

    @staticmethod
    def resolve_box_spec_for_sensor_pose(_spec, _index, _advcp_config, _ego_state, _lidar_pose):
        return np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float32)


def _patch_coperception_dependencies(monkeypatch):
    monkeypatch.setattr(_opencood_eval, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(ap_at_iou, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(mean_precision_at_iou, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(mean_recall_at_iou, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: DummyCommonUtils)
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", lambda: DummyCommonUtils)
    monkeypatch.setattr(attacker_benign_visibility_ratio, "_load_advcp_attack_helper", lambda: DummyAttackHelper)


def _box_tensor(count: int = 1):
    return np.zeros((count, 8, 3), dtype=np.float32)


def _memory_data():
    return OrderedDict(
        {
            0: OrderedDict(
                {
                    "ego": OrderedDict(
                        {
                            "ego": True,
                            "000001": {
                                "lidar_np": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                            },
                        }
                    ),
                    "cav-2": OrderedDict(
                        {
                            "000001": {
                                "lidar_np": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                            },
                        }
                    ),
                }
            )
        }
    )


def _valid_context():
    return {
        "pred_box_tensor": None,
        "pred_score": None,
        "gt_box_tensor": _box_tensor(),
        "advcp_config": {
            "boxes": [{"absolute": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
            "default_size": [2.0, 2.0, 2.0],
        },
        "memory_data": _memory_data(),
        "visualization_context": SimpleNamespace(
            mode="removal",
            removed_box_tensor=_box_tensor(),
            attacker_ids=["cav-2"],
        ),
    }


def _collector_with_all_coperception_metrics(warmup_steps: int):
    return MetricCollector(
        module="coperception",
        entity_id="global",
        metric_configs={
            "ap_at_iou": {"warmup_steps": warmup_steps},
            "mean_recall_at_iou": {"warmup_steps": warmup_steps},
            "mean_precision_at_iou": {"warmup_steps": warmup_steps},
            "attack_success_rate": {"warmup_steps": warmup_steps},
            "attacker_benign_visibility_ratio": {"warmup_steps": warmup_steps},
            "attacker_target_confidence": {"warmup_steps": warmup_steps},
        },
    )


def _metric_report(report, metric_name: str):
    for metric in report.metrics:
        if metric.metric_name == metric_name:
            return metric
    raise AssertionError(f"Metric {metric_name!r} was not reported.")


def test_all_coperception_metrics_exclude_warmup_samples_from_export_and_report(monkeypatch):
    _patch_coperception_dependencies(monkeypatch)
    collector = _collector_with_all_coperception_metrics(warmup_steps=2)
    context = _valid_context()

    collector.update(context)
    collector.update(context)
    collector.update(context)

    raw = collector.get_raw()
    for series in raw.series:
        assert all(sample.tick > 2 for sample in series.samples), series.name

    expected_single_sample_series = {
        "ap_iou_0_3",
        "ap_iou_0_5",
        "ap_iou_0_7",
        "mrec_iou_0_3",
        "mrec_iou_0_5",
        "mrec_iou_0_7",
        "mpre_iou_0_3",
        "mpre_iou_0_5",
        "mpre_iou_0_7",
        "asr_removal",
        "attacker_benign_visibility_ratio",
        "attacker_points_on_target",
        "benign_points_on_target",
        "confidence_removal",
    }
    for series_name in expected_single_sample_series:
        samples = raw.get_series(series_name)
        assert len(samples) == 1, series_name
        assert samples[0].tick == 3

    assert raw.get_series("asr_spoofing") == ()
    assert raw.get_series("confidence_spoofing") == ()

    report = UniversalReportBuilder().build_entity_report(raw)
    for metric_name in raw.active_metrics:
        metric_report = _metric_report(report, metric_name)
        assert any(summary.count == 1 for summary in metric_report.summary), metric_name
        assert all(summary.count in {0, 1} for summary in metric_report.summary), metric_name
