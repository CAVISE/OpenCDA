import numpy as np

from opencda.core.attack.advcp.types import AdvCPVisualizationContext
from opencda.metrics_tools.metrics import attack_success_rate
from opencda.metrics_tools.metrics.attack_success_rate import AttackSuccessRateMetric


class DummyPolygon:
    @property
    def centroid(self):
        return self

    def contains(self, _):
        return True


class OutsideZonePolygon:
    @property
    def centroid(self):
        return self

    def contains(self, _):
        return False


class DummyCommonUtils:
    @staticmethod
    def convert_format(boxes):
        return [DummyPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.ones(len(pred_polygon_list), dtype=np.float32)


class _DummyArea:
    def __init__(self, area):
        self.area = area


class OverlappingPolygon(DummyPolygon):
    """Polygon with full overlap — used to test deduplication (IoU=1.0 with any other polygon)."""

    def union(self, _):
        return _DummyArea(1.0)

    def intersection(self, _):
        return _DummyArea(1.0)


class DeduplicatingDummyCommonUtils:
    @staticmethod
    def convert_format(boxes):
        return [OverlappingPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.ones(len(pred_polygon_list), dtype=np.float32)


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


def _series_values(metric: AttackSuccessRateMetric, series_name: str) -> list[float]:
    for series in metric.get_raw():
        if series.name == series_name:
            return [sample.value for sample in series.samples]
    raise AssertionError(f"Series {series_name!r} was not exported.")


def test_removal_asr_succeeds_when_target_is_not_detected(monkeypatch):
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: DummyCommonUtils)
    metric = AttackSuccessRateMetric()

    metric.update(
        {
            "pred_box_tensor": None,
            "gt_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    assert _series_values(metric, "asr_removal") == [1.0]
    assert _series_values(metric, "asr_spoofing") == []


def test_removal_asr_fails_when_target_is_still_detected(monkeypatch):
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: DummyCommonUtils)
    metric = AttackSuccessRateMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "gt_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    assert _series_values(metric, "asr_removal") == [0.0]


def test_removal_asr_skips_when_no_gt_in_removal_zone(monkeypatch):
    class EmptyZoneUtils:
        @staticmethod
        def convert_format(boxes):
            return [OutsideZonePolygon() for _ in boxes]

        @staticmethod
        def compute_iou(_, pred_list):
            return np.zeros(len(pred_list), dtype=np.float32)

    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: EmptyZoneUtils)
    metric = AttackSuccessRateMetric()

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "gt_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    assert _series_values(metric, "asr_removal") == []


def test_removal_asr_reports_fraction_of_removed_targets(monkeypatch):
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", _make_first_hit_utils)
    metric = AttackSuccessRateMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "gt_box_tensor": _target_boxes(3),
            "visualization_context": AdvCPVisualizationContext(mode="removal", removed_box_tensor=_target_box()),
        }
    )

    # 1 GT target still detected, 2 removed → ASR = 2/3
    assert _series_values(metric, "asr_removal") == [2 / 3]


def test_spoofing_asr_succeeds_when_fake_target_is_detected(monkeypatch):
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: DummyCommonUtils)
    metric = AttackSuccessRateMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="spoofing", fake_box_tensor=_target_box()),
        }
    )

    assert _series_values(metric, "asr_spoofing") == [1.0]


def test_spoofing_asr_deduplicates_identical_fake_boxes(monkeypatch):
    """5 identical attacker boxes (N attackers, same target) → 1 unique target after dedup."""
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: DeduplicatingDummyCommonUtils)
    metric = AttackSuccessRateMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="spoofing", fake_box_tensor=_target_boxes(5)),
        }
    )

    values = _series_values(metric, "asr_spoofing")
    assert len(values) == 1
    assert values[0] == 1.0


def test_asr_skips_frames_without_attack_target():
    metric = AttackSuccessRateMetric()

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "visualization_context": AdvCPVisualizationContext(mode="spoofing", fake_box_tensor="test-placeholder"),
        }
    )

    assert _series_values(metric, "asr_removal") == []
    assert _series_values(metric, "asr_spoofing") == []
