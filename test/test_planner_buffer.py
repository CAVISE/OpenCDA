"""Regression coverage for route buffers exhausted before path generation."""

import importlib
import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opencda.core.plan.local_planner_behavior import LocalPlanner


@pytest.fixture
def behavior_type():
    name = "opencda.core.plan.behavior_agent"
    placeholder = sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name).BehaviorAgent
    finally:
        sys.modules.pop(name, None)
        if placeholder is not None:
            sys.modules[name] = placeholder


def planner():
    local = LocalPlanner.__new__(LocalPlanner)
    local._buffer_size = 3
    local.waypoint_update_freq = 2
    local._waypoint_buffer = deque(maxlen=3)
    local.waypoints_queue = deque()
    local._trajectory_buffer = deque(["stale trajectory"])
    return local


def agent(behavior_type, local):
    behavior = behavior_type.__new__(behavior_type)
    behavior._local_planner = local
    behavior._ego_pos = SimpleNamespace(location=object())
    behavior._map = Mock()
    behavior.vehicle = SimpleNamespace(id=200)
    behavior.objects = {}
    behavior.overtake_counter = 0
    behavior.destination_push_flag = 0
    behavior.overtake_allowed_origin = True
    behavior.debug = False
    behavior.end_waypoint = Mock()
    behavior.is_intersection = Mock(return_value=False)
    behavior.is_close_to_destination = Mock(return_value=False)
    behavior.traffic_light_manager = Mock(return_value=0)
    behavior.set_destination = Mock()
    return behavior


def test_empty_path_does_not_access_map():
    local = planner()
    assert local.generate_path() == ([], [], [], [])


def test_refill_preserves_order_capacity_and_threshold():
    local = planner()
    local.waypoints_queue.extend(range(5))
    local.refill_waypoint_buffer()
    assert list(local._waypoint_buffer) == [0, 1, 2]
    assert list(local.waypoints_queue) == [3, 4]
    local._waypoint_buffer.popleft()
    local.refill_waypoint_buffer()
    assert list(local._waypoint_buffer) == [1, 2]
    assert list(local.waypoints_queue) == [3, 4]


def test_filter_exhaustion_refills_before_path_generation(behavior_type):
    local = planner()
    local.waypoints_queue.extend(range(8))
    local.buffer_filter = Mock(side_effect=local._waypoint_buffer.clear)
    behavior = agent(behavior_type, local)

    class PathReached(Exception):
        pass

    def check_path():
        assert list(local._waypoint_buffer) == [3, 4, 5]
        assert list(local.waypoints_queue) == [6, 7]
        raise PathReached

    local.generate_path = check_path
    with pytest.raises(PathReached):
        behavior.run_step()
    behavior.set_destination.assert_not_called()


def test_empty_rebuilt_route_stops_and_discards_stale_trajectory(behavior_type, caplog):
    local = planner()
    local.buffer_filter = Mock()
    behavior = agent(behavior_type, local)
    assert behavior.run_step() == (0, None)
    behavior.set_destination.assert_called_once()
    assert not local.get_trajectory()
    assert "Vehicle 200 has no planned path" in caplog.text
