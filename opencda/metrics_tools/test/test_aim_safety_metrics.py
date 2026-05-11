from opencda.metrics_tools.metrics.collision_count import CollisionCountMetric
from opencda.metrics_tools.metrics.hard_brake_count import HardBrakeCountMetric
from opencda.metrics_tools.metrics.identity_conflict_count import IdentityConflictCountMetric
from opencda.metrics_tools.metrics.near_miss_count import NearMissCountMetric


def _series_values(metric, series_name: str) -> list[float]:
    for series in metric.get_raw():
        if series.name == series_name:
            return [sample.value for sample in series.samples]
    return []


def test_hard_brake_count_counts_episodes_not_ticks():
    metric = HardBrakeCountMetric(warmup_steps=0, dt=1.0, deceleration_threshold=-3.0, reset_threshold=-1.0)

    for speed_kmh in (36.0, 18.0, 10.8, 18.0, 0.0):
        metric.update({"ego_speed": speed_kmh})

    assert _series_values(metric, "hard_brake_count") == [0.0, 1.0, 1.0, 1.0, 2.0]


def test_collision_count_counts_each_collided_vehicle_once():
    metric = CollisionCountMetric(warmup_steps=0)

    metric.update({"vehicles": ({"node_id": "cav-1", "collided": True}, {"node_id": "cav-2", "collided": False})})
    metric.update({"vehicles": ({"node_id": "cav-1", "collided": True}, {"node_id": "cav-2", "collided": False})})
    metric.update({"vehicles": ({"node_id": "cav-1", "collided": True}, {"node_id": "cav-2", "collided": True})})

    assert _series_values(metric, "collision_count") == [1.0, 1.0, 2.0]


def test_near_miss_count_counts_pair_episodes():
    metric = NearMissCountMetric(warmup_steps=0, distance_threshold=2.0)

    metric.update({"vehicles": ({"node_id": "cav-1", "x": 0.0, "y": 0.0, "z": 0.0}, {"node_id": "cav-2", "x": 1.0, "y": 0.0, "z": 0.0})})
    metric.update({"vehicles": ({"node_id": "cav-1", "x": 0.0, "y": 0.0, "z": 0.0}, {"node_id": "cav-2", "x": 1.5, "y": 0.0, "z": 0.0})})
    metric.update({"vehicles": ({"node_id": "cav-1", "x": 0.0, "y": 0.0, "z": 0.0}, {"node_id": "cav-2", "x": 5.0, "y": 0.0, "z": 0.0})})
    metric.update({"vehicles": ({"node_id": "cav-1", "x": 0.0, "y": 0.0, "z": 0.0}, {"node_id": "cav-2", "x": 1.0, "y": 0.0, "z": 0.0})})

    assert _series_values(metric, "near_miss_count") == [1.0, 1.0, 1.0, 2.0]


def test_identity_conflict_count_counts_conflict_episodes():
    metric = IdentityConflictCountMetric(warmup_steps=0)

    metric.update({"identity_claims": ({"producer_node_id": "cav-1", "claimed_node_id": "cav-1"},)})
    metric.update(
        {
            "identity_claims": (
                {"producer_node_id": "cav-1", "claimed_node_id": "cav-1"},
                {"producer_node_id": "cav-2", "claimed_node_id": "cav-1"},
            )
        }
    )
    metric.update(
        {
            "identity_claims": (
                {"producer_node_id": "cav-1", "claimed_node_id": "cav-1"},
                {"producer_node_id": "cav-2", "claimed_node_id": "cav-1"},
            )
        }
    )
    metric.update({"identity_claims": ({"producer_node_id": "cav-1", "claimed_node_id": "cav-1"},)})
    metric.update(
        {
            "identity_claims": (
                {"producer_node_id": "cav-1", "claimed_node_id": "cav-1"},
                {"producer_node_id": "cav-2", "claimed_node_id": "cav-1"},
            )
        }
    )

    assert _series_values(metric, "identity_conflict_count") == [0.0, 1.0, 1.0, 1.0, 2.0]
