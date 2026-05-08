from types import SimpleNamespace

import numpy as np

from opencda.metrics_tools.metrics import attack_success_rate
from opencda.metrics_tools.metrics.attack_success_rate import AttackSuccessRateMetric


class DummyPolygon:
    @property
    def centroid(self):
        return self

    def covers(self, _):
        return True


class DummyCommonUtils:
    @staticmethod
    def convert_format(boxes):
        return [DummyPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.ones(len(pred_polygon_list), dtype=np.float32)


class AlternatingCoverPolygon:
    _cover_results = [True, False, False]

    def __init__(self):
        self.cover_result = self._cover_results.pop(0)

    @property
    def centroid(self):
        return self

    def covers(self, _):
        return self.cover_result


class PartialRemovalCommonUtils:
    @staticmethod
    def convert_format(boxes):
        AlternatingCoverPolygon._cover_results = [True, False, False]
        return [AlternatingCoverPolygon() for _ in boxes]

    @staticmethod
    def compute_iou(_, pred_polygon_list):
        return np.zeros(len(pred_polygon_list), dtype=np.float32)


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
            "visualization_context": {
                "mode": "removal",
                "removed_box_tensor": _target_box(),
            },
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
            "visualization_context": SimpleNamespace(
                mode="removal",
                removed_box_tensor=_target_box(),
            ),
        }
    )

    assert _series_values(metric, "asr_removal") == [0.0]


def test_removal_asr_reports_fraction_of_removed_targets(monkeypatch):
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: PartialRemovalCommonUtils)
    metric = AttackSuccessRateMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "visualization_context": {
                "mode": "removal",
                "removed_box_tensor": _target_boxes(3),
            },
        }
    )

    assert _series_values(metric, "asr_removal") == [2 / 3]


def test_spoofing_asr_succeeds_when_fake_target_is_detected(monkeypatch):
    monkeypatch.setattr(attack_success_rate, "_load_common_utils", lambda: DummyCommonUtils)
    metric = AttackSuccessRateMetric(iou_threshold=0.3)

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "visualization_context": {
                "mode": "spoofing",
                "fake_box_tensor": _target_box(),
            },
        }
    )

    assert _series_values(metric, "asr_spoofing") == [1.0]


def test_asr_skips_frames_without_attack_target():
    metric = AttackSuccessRateMetric()

    metric.update(
        {
            "pred_box_tensor": _target_box(),
            "visualization_context": {
                "mode": "spoofing",
                "fake_box_tensor": "test-placeholder",
            },
        }
    )

    assert _series_values(metric, "asr_removal") == []
    assert _series_values(metric, "asr_spoofing") == []
