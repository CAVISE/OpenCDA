from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from opencda.metrics_tools.metrics import attacker_benign_visibility_ratio
from opencda.metrics_tools.metrics.attacker_benign_visibility_ratio import AttackerBenignVisibilityRatioMetric


class DummyAttackHelper:
    @staticmethod
    def resolve_ego_agent_id(scenario_data):
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
                                "lidar_np": np.array(
                                    [
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.5, 0.5, 1.0, 1.0],
                                        [5.0, 5.0, 1.0, 1.0],
                                    ],
                                    dtype=np.float32,
                                ),
                            },
                        }
                    ),
                    "cav-3": OrderedDict(
                        {
                            "000001": {
                                "lidar_np": np.array(
                                    [
                                        [0.2, 0.2, 1.0, 1.0],
                                        [6.0, 6.0, 1.0, 1.0],
                                    ],
                                    dtype=np.float32,
                                ),
                            },
                        }
                    ),
                }
            )
        }
    )


def _series_values(metric: AttackerBenignVisibilityRatioMetric, series_name: str) -> list[float]:
    for series in metric.get_raw():
        if series.name == series_name:
            return [sample.value for sample in series.samples]
    raise AssertionError(f"Series {series_name!r} was not exported.")


def test_attacker_benign_visibility_ratio_counts_target_points(monkeypatch):
    monkeypatch.setattr(attacker_benign_visibility_ratio, "_load_advcp_attack_helper", lambda: DummyAttackHelper)
    metric = AttackerBenignVisibilityRatioMetric(epsilon=1.0)

    metric.update(
        {
            "advcp_config": {
                "boxes": [{"absolute": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
                "default_size": [2.0, 2.0, 2.0],
            },
            "memory_data": _memory_data(),
            "visualization_context": SimpleNamespace(mode="remove", attacker_ids=["cav-2"]),
        }
    )

    assert _series_values(metric, "attacker_points_on_target") == [2.0]
    assert _series_values(metric, "benign_points_on_target") == [2.0]
    assert _series_values(metric, "attacker_benign_visibility_ratio") == [2.0 / 3.0]


def test_attacker_benign_visibility_ratio_skips_without_attack_context(monkeypatch):
    monkeypatch.setattr(attacker_benign_visibility_ratio, "_load_advcp_attack_helper", lambda: DummyAttackHelper)
    metric = AttackerBenignVisibilityRatioMetric(epsilon=1.0)

    metric.update(
        {
            "advcp_config": {"boxes": [{"absolute": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]},
            "memory_data": _memory_data(),
            "visualization_context": SimpleNamespace(mode="remove", attacker_ids=[]),
        }
    )

    assert _series_values(metric, "attacker_benign_visibility_ratio") == []
