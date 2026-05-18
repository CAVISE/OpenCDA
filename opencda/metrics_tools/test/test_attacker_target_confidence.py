import numpy as np

from opencda.core.attack.advcp.types import AdvCPVisualizationContext
from opencda.metrics_tools.metrics.coperception import attacker_target_confidence
from opencda.metrics_tools.metrics.coperception.attacker_target_confidence import AttackerTargetConfidenceMetric


class DummyPolygon:
    @property
    def centroid(self):
        return self

    def contains(self, _):
        return True


class HighIoUCommonUtils:
    @staticmethod
    def convert_format(boxes):
        return [DummyPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.full(len(pred_polygon_list), 0.9, dtype=np.float32)


class LowIoUCommonUtils:
    @staticmethod
    def convert_format(boxes):
        return [DummyPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.zeros(len(pred_polygon_list), dtype=np.float32)


def _make_first_hit_utils():
    """compute_iou returns ones for the first call only (first GT target still detected)."""
    call_count = [0]

    class FirstHitCommonUtils:
        @staticmethod
        def convert_format(boxes):
            return [DummyPolygon() for _ in boxes]

        @staticmethod
        def compute_iou(_, pred_list):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.ones(len(pred_list), dtype=np.float32)
            return np.zeros(len(pred_list), dtype=np.float32)

    return FirstHitCommonUtils


def _target_box():
    return np.zeros((1, 8, 3), dtype=np.float32)


def _target_boxes(count: int):
    return np.zeros((count, 8, 3), dtype=np.float32)


def _series_values(metric: AttackerTargetConfidenceMetric, series_name: str) -> list[float]:
    for series in metric.get_raw():
        if series.name == series_name:
            return [sample.value for sample in series.samples]
    raise AssertionError(f"Series {series_name!r} was not exported.")


def test_removal_confidence_zero_when_target_is_not_detected(monkeypatch):
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", lambda: LowIoUCommonUtils)
    metric = AttackerTargetConfidenceMetric()

    metric.update(
        {
            "pred_box_tensor": None,
            "pred_score": None,
            "gt_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    assert _series_values(metric, "confidence_removal") == [0.0]
    assert _series_values(metric, "confidence_spoofing") == []


def test_confidence_respects_warmup_steps(monkeypatch):
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", lambda: LowIoUCommonUtils)
    metric = AttackerTargetConfidenceMetric(warmup_steps=1)
    context = {
        "pred_box_tensor": None,
        "pred_score": None,
        "gt_box_tensor": _target_box(),
        "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
    }

    metric.update(context)
    assert _series_values(metric, "confidence_removal") == []

    metric.update(context)

    assert _series_values(metric, "confidence_removal") == [0.0]


def test_removal_confidence_uses_iou_match_for_score(monkeypatch):
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", lambda: HighIoUCommonUtils)
    metric = AttackerTargetConfidenceMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_boxes(3),
            "pred_score": np.array([0.8, 0.5, 0.4], dtype=np.float32),
            "gt_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    values = _series_values(metric, "confidence_removal")
    assert len(values) == 1
    assert abs(values[0] - 0.8) < 1e-5


def test_removal_confidence_averages_per_target(monkeypatch):
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", _make_first_hit_utils)
    metric = AttackerTargetConfidenceMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_boxes(3),
            "pred_score": np.array([0.6, 0.0, 0.0], dtype=np.float32),
            "gt_box_tensor": _target_boxes(3),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    # first GT target: pred[0] best IoU match → score 0.6; others: no match → 0.0
    values = _series_values(metric, "confidence_removal")
    assert len(values) == 1
    assert abs(values[0] - 0.6 / 3) < 1e-5


def test_spoofing_confidence_uses_iou_match(monkeypatch):
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", lambda: HighIoUCommonUtils)
    metric = AttackerTargetConfidenceMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "pred_score": np.array([0.7], dtype=np.float32),
            "visualization_context": AdvCPVisualizationContext(mode="spoofing", fake_box_tensor=_target_box()),
        }
    )

    values = _series_values(metric, "confidence_spoofing")
    assert len(values) == 1
    assert abs(values[0] - 0.7) < 1e-5


def test_spoofing_confidence_zero_when_iou_below_threshold(monkeypatch):
    monkeypatch.setattr(attacker_target_confidence, "_load_common_utils", lambda: LowIoUCommonUtils)
    metric = AttackerTargetConfidenceMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "pred_score": np.array([0.9], dtype=np.float32),
            "visualization_context": AdvCPVisualizationContext(mode="spoofing", fake_box_tensor=_target_box()),
        }
    )

    assert _series_values(metric, "confidence_spoofing") == [0.0]


def test_metric_skips_frames_without_attack_target():
    metric = AttackerTargetConfidenceMetric()

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "pred_score": np.array([0.5], dtype=np.float32),
            "visualization_context": AdvCPVisualizationContext(mode="spoofing", fake_box_tensor="test-placeholder"),
        }
    )

    assert _series_values(metric, "confidence_removal") == []
    assert _series_values(metric, "confidence_spoofing") == []
