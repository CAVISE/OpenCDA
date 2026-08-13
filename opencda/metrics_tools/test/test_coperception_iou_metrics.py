import numpy as np

from opencda.metrics_tools.metrics.coperception import (
    _opencood_eval,
    ap_at_iou,
    mean_precision_at_iou,
    mean_recall_at_iou,
)
from opencda.metrics_tools.metrics.coperception.ap_at_iou import APAtIoUMetric
from opencda.metrics_tools.metrics.coperception.mean_precision_at_iou import MeanPrecisionAtIoUMetric
from opencda.metrics_tools.metrics.coperception.mean_recall_at_iou import MeanRecallAtIoUMetric


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


def _patch_eval_utils(monkeypatch):
    monkeypatch.setattr(_opencood_eval, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(ap_at_iou, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(mean_precision_at_iou, "load_eval_utils", lambda: DummyEvalUtils)
    monkeypatch.setattr(mean_recall_at_iou, "load_eval_utils", lambda: DummyEvalUtils)


def _context():
    return {
        "pred_box_tensor": np.zeros((1, 8, 3), dtype=np.float32),
        "pred_score": np.ones((1,), dtype=np.float32),
        "gt_box_tensor": np.zeros((1, 8, 3), dtype=np.float32),
    }


def _series(metric, series_name):
    for series in metric.get_raw():
        if series.name == series_name:
            return series.samples
    raise AssertionError(f"Series {series_name!r} was not exported.")


def test_ap_at_iou_respects_warmup_steps(monkeypatch):
    _patch_eval_utils(monkeypatch)
    metric = APAtIoUMetric(warmup_steps=2)

    metric.update(_context())
    metric.update(_context())
    assert _series(metric, "ap_iou_0_3") == ()

    metric.update(_context())

    samples = _series(metric, "ap_iou_0_3")
    assert len(samples) == 1
    assert samples[0].tick == 3
    assert samples[0].value == 0.5


def test_mean_precision_at_iou_exports_one_sample_per_valid_update(monkeypatch):
    _patch_eval_utils(monkeypatch)
    metric = MeanPrecisionAtIoUMetric(warmup_steps=1)

    metric.update(_context())
    metric.update(_context())

    samples = _series(metric, "mpre_iou_0_3")
    assert len(samples) == 1
    assert samples[0].tick == 2
    assert abs(samples[0].value - 0.7) < 1e-9


def test_mean_recall_at_iou_exports_one_sample_per_valid_update(monkeypatch):
    _patch_eval_utils(monkeypatch)
    metric = MeanRecallAtIoUMetric(warmup_steps=1)

    metric.update(_context())
    metric.update(_context())

    samples = _series(metric, "mrec_iou_0_3")
    assert len(samples) == 1
    assert samples[0].tick == 2
    assert abs(samples[0].value - 0.5) < 1e-9
