"""
Contract unit tests for opencda.core.sensing.perception.o3d_lidar_libs.

These tests:
- Do not require real Open3D (a minimal in-test stub is installed into sys.modules["open3d"]).
- Avoid GUI and timing flakiness (time.sleep is patched).
- Avoid depending on SciPy mode implementation details (mode is patched deterministically).
- Focus on contract behaviors and branch coverage.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore:A NumPy version .* is required for this version of SciPy.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Unable to import Axes3D.*:UserWarning"),
]


class _Vector3dVector:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class _AxisAlignedBoundingBox:
    def __init__(self, min_bound=None, max_bound=None):
        self.min_bound = np.asarray(min_bound, dtype=float) if min_bound is not None else None
        self.max_bound = np.asarray(max_bound, dtype=float) if max_bound is not None else None
        self.color = None

    def get_box_points(self) -> np.ndarray:
        x0, y0, z0 = self.min_bound
        x1, y1, z1 = self.max_bound
        return np.array(
            [
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1],
            ],
            dtype=float,
        )


class _PointCloud:
    def __init__(self):
        self.points = None
        self.colors = None

    def get_axis_aligned_bounding_box(self) -> _AxisAlignedBoundingBox:
        pts = self.points.data if isinstance(self.points, _Vector3dVector) else np.asarray(self.points, dtype=float)
        min_bound = np.min(pts, axis=0)
        max_bound = np.max(pts, axis=0)
        return _AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)


class _RenderOption:
    def __init__(self):
        self.background_color = None
        self.point_size = None
        self.show_coordinate_frame = None


class _Visualizer:
    def __init__(self):
        self._render_option = _RenderOption()
        self.created_window = None
        self.added = []
        self.removed = []
        self.updated = []

        self.poll_events_called = 0
        self.update_renderer_called = 0

    def create_window(self, **kwargs):
        self.created_window = kwargs

    def get_render_option(self):
        return self._render_option

    def add_geometry(self, geom):
        self.added.append(geom)

    def remove_geometry(self, geom):
        self.removed.append(geom)

    def update_geometry(self, geom):
        self.updated.append(geom)

    def poll_events(self):
        self.poll_events_called += 1

    def update_renderer(self):
        self.update_renderer_called += 1


@pytest.fixture
def open3d_stub(monkeypatch):
    """
    Install a minimal Open3D stub into sys.modules["open3d"].

    test/conftest.py may already create an empty module named "open3d";
    this fixture ensures required attributes exist.
    """
    import sys

    o3d = sys.modules["open3d"]

    if not hasattr(o3d, "utility"):
        o3d.utility = types.SimpleNamespace()
    if not hasattr(o3d, "geometry"):
        o3d.geometry = types.SimpleNamespace()
    if not hasattr(o3d, "visualization"):
        o3d.visualization = types.SimpleNamespace()

    monkeypatch.setattr(o3d.utility, "Vector3dVector", _Vector3dVector, raising=False)
    monkeypatch.setattr(o3d.geometry, "PointCloud", _PointCloud, raising=False)
    monkeypatch.setattr(o3d.geometry, "AxisAlignedBoundingBox", _AxisAlignedBoundingBox, raising=False)
    monkeypatch.setattr(o3d.visualization, "Visualizer", _Visualizer, raising=False)

    return o3d


def test_o3d_pointcloud_encode_sets_points_colors_and_inverts_first_axis(open3d_stub):
    from opencda.core.sensing.perception.o3d_lidar_libs import o3d_pointcloud_encode

    raw = np.array(
        [
            [1.0, 10.0, 0.5, 1.0],
            [2.0, 11.0, 0.6, 2.0],
            [-3.0, 12.0, 0.7, 3.0],
        ],
        dtype=float,
    )
    pc = open3d_stub.geometry.PointCloud()

    o3d_pointcloud_encode(raw, pc)

    assert isinstance(pc.points, _Vector3dVector)
    assert isinstance(pc.colors, _Vector3dVector)

    pts = pc.points.data
    assert pts.shape == (3, 3)

    assert pts[:, 0] == pytest.approx(np.array([-1.0, -2.0, 3.0], dtype=float))
    assert pts[:, 1] == pytest.approx(np.array([10.0, 11.0, 12.0], dtype=float))
    assert pts[:, 2] == pytest.approx(np.array([0.5, 0.6, 0.7], dtype=float))

    cols = pc.colors.data
    assert cols.shape == (3, 3)
    assert np.all(cols >= 0.0)
    assert np.all(cols <= 1.0)

    assert np.isfinite(cols).all()


def test_o3d_pointcloud_encode_does_not_mutate_raw_data_and_color_changes_with_intensity(open3d_stub):
    from opencda.core.sensing.perception.o3d_lidar_libs import o3d_pointcloud_encode

    # Choose intensities that map to intensity_col within (0, 1) range:
    # intensity_col = 1 - log(intensity) / log(exp(-0.004 * 100)) = 1 + log(intensity) / 0.4
    # intensity in (exp(-0.4), 1) => intensity_col in (0, 1)
    raw = np.array(
        [
            [1.0, 10.0, 0.5, 0.7],
            [2.0, 11.0, 0.6, 0.8],
            [3.0, 12.0, 0.7, 0.9],
        ],
        dtype=float,
    )
    raw_before = raw.copy()

    pc = open3d_stub.geometry.PointCloud()
    o3d_pointcloud_encode(raw, pc)

    assert raw == pytest.approx(raw_before)

    intensities = raw[:, -1]
    denom = np.log(np.exp(-0.004 * 100))
    intensity_col = 1.0 - np.log(intensities) / denom
    assert np.all(intensity_col > 0.0)
    assert np.all(intensity_col < 1.0)
    assert np.all(np.diff(intensity_col) > 0.0)

    cols = pc.colors.data
    assert cols.shape == (3, 3)
    assert np.isfinite(cols).all()

    d01 = float(np.max(np.abs(cols[0] - cols[1])))
    d12 = float(np.max(np.abs(cols[1] - cols[2])))
    assert d01 > 1e-9
    assert d12 > 1e-9


def test_o3d_visualizer_init_sets_window_and_render_options(open3d_stub):
    from opencda.core.sensing.perception.o3d_lidar_libs import o3d_visualizer_init

    vis = o3d_visualizer_init(actor_id=42)

    assert isinstance(vis, _Visualizer)
    assert vis.created_window == {
        "window_name": "42",
        "width": 480,
        "height": 320,
        "left": 480,
        "top": 270,
    }

    ro = vis.get_render_option()
    assert ro.background_color == [0.05, 0.05, 0.05]
    assert ro.point_size == 1
    assert ro.show_coordinate_frame is True


def test_o3d_visualizer_show_adds_pointcloud_only_on_count_2_and_draws_vehicles(open3d_stub, monkeypatch):
    from opencda.core.sensing.perception import o3d_lidar_libs as libs

    monkeypatch.setattr(libs.time, "sleep", lambda _t: None)

    vis = open3d_stub.visualization.Visualizer()
    pc = open3d_stub.geometry.PointCloud()

    class _Veh:
        def __init__(self, bbx):
            self.o3d_bbx = bbx

    bbx1 = open3d_stub.geometry.AxisAlignedBoundingBox(min_bound=[0, 0, 0], max_bound=[1, 1, 1])
    bbx2 = open3d_stub.geometry.AxisAlignedBoundingBox(min_bound=[2, 2, 2], max_bound=[3, 3, 3])

    objects = {"vehicles": [_Veh(bbx1), _Veh(bbx2)], "static": [object()]}

    libs.o3d_visualizer_show(vis, count=2, point_cloud=pc, objects=objects)

    assert vis.added.count(pc) == 1
    assert bbx1 in vis.added and bbx2 in vis.added
    assert len(vis.added) == 3

    assert bbx1 in vis.removed and bbx2 in vis.removed
    assert len(vis.removed) == 2

    assert vis.updated[-1] is pc
    assert vis.poll_events_called == 1
    assert vis.update_renderer_called == 1

    vis2 = open3d_stub.visualization.Visualizer()
    libs.o3d_visualizer_show(vis2, count=3, point_cloud=pc, objects=objects)
    assert pc not in vis2.added


def test_o3d_visualizer_show_handles_missing_vehicles_key(open3d_stub, monkeypatch):
    from opencda.core.sensing.perception import o3d_lidar_libs as libs

    monkeypatch.setattr(libs.time, "sleep", lambda _t: None)

    vis = open3d_stub.visualization.Visualizer()
    pc = open3d_stub.geometry.PointCloud()
    objects = {"static": [object()]}

    libs.o3d_visualizer_show(vis, count=2, point_cloud=pc, objects=objects)

    assert vis.added == [pc]
    assert vis.removed == []
    assert vis.updated == [pc]
    assert vis.poll_events_called == 1
    assert vis.update_renderer_called == 1


class _FakeTorchTensor:
    def __init__(self, arr: np.ndarray, *, is_cuda: bool):
        self._arr = np.asarray(arr)
        self.is_cuda = is_cuda
        self.cpu_called = 0
        self.detach_called = 0

    def cpu(self):
        self.cpu_called += 1
        return self

    def detach(self):
        self.detach_called += 1
        return self

    def numpy(self):
        return self._arr


class _FakeLidarSensor:
    def __init__(self, transform):
        self._transform = transform

    def get_transform(self):
        return self._transform


def _stable_mode_int(arr, axis=0):  # noqa: ARG001
    """
    +    Deterministic replacement for scipy.stats.mode used by this unit test.
    +
    +    Returns a tuple compatible with the production usage: mode(...)[0][0].
    +    In case of ties, returns the smallest value to keep behavior deterministic.
    +"""
    arr = np.asarray(arr).astype(int, copy=False).ravel()
    values, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    candidates = values[counts == max_count]
    mode_value = int(candidates.min())

    return (np.array([mode_value], dtype=int), None)


def test_o3d_camera_lidar_fusion_vehicle_and_static_branches(open3d_stub, monkeypatch):
    import carla

    from opencda.core.sensing.perception import o3d_lidar_libs as libs
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle
    from opencda.core.sensing.perception.static_obstacle import StaticObstacle

    monkeypatch.setattr(libs.st, "sensor_to_world", lambda corner, _tf: corner)
    monkeypatch.setattr(libs, "mode", _stable_mode_int)

    lidar_sensor = _FakeLidarSensor(transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0)))

    lidar_3d = np.array(
        [
            [5.1, 1.0, 0.5, 10.0],
            [5.5, 1.2, 0.6, 11.0],
            [5.2, 1.1, 0.7, 12.0],
        ],
        dtype=float,
    )
    projected = np.array(
        [
            [50.0, 50.0, 1.0],
            [60.0, 60.0, 1.0],
            [55.0, 55.0, 1.0],
        ],
        dtype=float,
    )

    yolo_arr = np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 1.0]], dtype=float)
    yolo_bbx = _FakeTorchTensor(yolo_arr, is_cuda=True)

    objects = {"vehicles": [], "traffic_lights": []}
    out = libs.o3d_camera_lidar_fusion(objects, yolo_bbx, lidar_3d, projected, lidar_sensor)

    assert yolo_bbx.cpu_called == 1
    assert yolo_bbx.detach_called == 1

    assert "vehicles" in out
    assert len(out["vehicles"]) == 1
    assert isinstance(out["vehicles"][0], ObstacleVehicle)
    assert out["vehicles"][0].o3d_bbx.color == (0, 1, 0)

    yolo_arr_static = np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 0.0]], dtype=float)
    yolo_bbx_static = _FakeTorchTensor(yolo_arr_static, is_cuda=False)

    objects2 = {"vehicles": [], "traffic_lights": []}
    out2 = libs.o3d_camera_lidar_fusion(objects2, yolo_bbx_static, lidar_3d, projected, lidar_sensor)

    assert yolo_bbx_static.cpu_called == 0
    assert yolo_bbx_static.detach_called == 1

    assert "static" in out2
    assert len(out2["static"]) == 1
    assert isinstance(out2["static"][0], StaticObstacle)
    assert out2["static"][0].o3d_bbx.color == (0, 1, 0)


def test_o3d_camera_lidar_fusion_creates_keys_when_missing_and_accumulates(open3d_stub, monkeypatch):
    import carla

    from opencda.core.sensing.perception import o3d_lidar_libs as libs

    monkeypatch.setattr(libs, "mode", _stable_mode_int)

    captured = {"sensor_to_world_calls": 0}

    def _sensor_to_world_assert_shape(corner, _tf):
        assert isinstance(corner, np.ndarray)
        assert corner.shape == (4, 8)
        captured["sensor_to_world_calls"] += 1
        return corner

    monkeypatch.setattr(libs.st, "sensor_to_world", _sensor_to_world_assert_shape)

    lidar_sensor = _FakeLidarSensor(transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0)))

    lidar_3d = np.array(
        [
            [5.0, 1.0, 0.5, 10.0],
            [5.4, 1.2, 0.6, 11.0],
            [5.2, 1.1, 0.7, 12.0],
        ],
        dtype=float,
    )
    projected = np.array(
        [
            [50.0, 50.0, 1.0],
            [60.0, 60.0, 1.0],
            [55.0, 55.0, 1.0],
        ],
        dtype=float,
    )

    yolo_vehicle = _FakeTorchTensor(np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 1.0]], dtype=float), is_cuda=False)
    yolo_static = _FakeTorchTensor(np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 0.0]], dtype=float), is_cuda=False)

    objects = {}
    out = libs.o3d_camera_lidar_fusion(objects, yolo_vehicle, lidar_3d, projected, lidar_sensor)
    out = libs.o3d_camera_lidar_fusion(out, yolo_static, lidar_3d, projected, lidar_sensor)
    out = libs.o3d_camera_lidar_fusion(out, yolo_vehicle, lidar_3d, projected, lidar_sensor)

    assert "vehicles" in out and "static" in out
    assert len(out["vehicles"]) == 2
    assert len(out["static"]) == 1
    assert captured["sensor_to_world_calls"] == 3


def test_o3d_camera_lidar_fusion_filters_outlier_and_requires_positive_depth(open3d_stub, monkeypatch):
    import carla

    from opencda.core.sensing.perception import o3d_lidar_libs as libs

    monkeypatch.setattr(libs, "mode", _stable_mode_int)
    monkeypatch.setattr(libs.st, "sensor_to_world", lambda corner, _tf: corner)

    lidar_sensor = _FakeLidarSensor(transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0)))

    lidar_3d = np.array(
        [
            [5.1, 1.0, 0.5, 10.0],
            [5.5, 1.2, 0.6, 11.0],
            [5.2, 1.1, 0.7, 12.0],
            [50.0, 50.0, 0.6, 9.0],
            [5.3, 1.05, 0.6, 8.0],
        ],
        dtype=float,
    )
    projected = np.array(
        [
            [50.0, 50.0, 1.0],
            [60.0, 60.0, 1.0],
            [55.0, 55.0, 1.0],
            [70.0, 70.0, 1.0],
            [52.0, 52.0, -1.0],
        ],
        dtype=float,
    )

    yolo_vehicle = _FakeTorchTensor(np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 1.0]], dtype=float), is_cuda=False)
    out = libs.o3d_camera_lidar_fusion({}, yolo_vehicle, lidar_3d, projected, lidar_sensor)

    assert len(out["vehicles"]) == 1
    aabb = out["vehicles"][0].o3d_bbx

    assert aabb.min_bound[0] == pytest.approx(-5.5)
    assert aabb.max_bound[0] == pytest.approx(-5.1)

    assert aabb.min_bound[1] == pytest.approx(1.0)
    assert aabb.max_bound[1] == pytest.approx(1.2)

    assert aabb.min_bound[2] == pytest.approx(0.5)
    assert aabb.max_bound[2] == pytest.approx(0.7)


def test_o3d_camera_lidar_fusion_skips_when_no_points_in_bbox(open3d_stub, monkeypatch):
    import carla

    from opencda.core.sensing.perception import o3d_lidar_libs as libs

    monkeypatch.setattr(libs.st, "sensor_to_world", lambda corner, _tf: corner)
    monkeypatch.setattr(libs, "mode", _stable_mode_int)

    lidar_sensor = _FakeLidarSensor(transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0)))

    lidar_3d = np.array([[5.0, 1.0, 0.5, 10.0]], dtype=float)
    projected = np.array([[500.0, 500.0, 1.0]], dtype=float)

    yolo_arr = np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 1.0]], dtype=float)
    yolo_bbx = _FakeTorchTensor(yolo_arr, is_cuda=False)

    objects = {"vehicles": [], "traffic_lights": []}
    out = libs.o3d_camera_lidar_fusion(objects, yolo_bbx, lidar_3d, projected, lidar_sensor)

    assert "vehicles" in out
    assert out["vehicles"] == []
    assert "static" not in out


def test_o3d_camera_lidar_fusion_skips_when_inliers_less_than_two(open3d_stub, monkeypatch):
    import carla

    from opencda.core.sensing.perception import o3d_lidar_libs as libs

    monkeypatch.setattr(libs.st, "sensor_to_world", lambda corner, _tf: corner)
    monkeypatch.setattr(libs, "mode", _stable_mode_int)

    lidar_sensor = _FakeLidarSensor(transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0)))

    lidar_3d = np.array(
        [
            [5.0, 1.0, 0.5, 10.0],
            [50.0, 50.0, 0.5, 10.0],
        ],
        dtype=float,
    )
    projected = np.array(
        [
            [10.0, 10.0, 1.0],
            [20.0, 20.0, 1.0],
        ],
        dtype=float,
    )

    yolo_arr = np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 1.0]], dtype=float)
    yolo_bbx = _FakeTorchTensor(yolo_arr, is_cuda=False)

    objects = {"vehicles": [], "traffic_lights": []}
    out = libs.o3d_camera_lidar_fusion(objects, yolo_bbx, lidar_3d, projected, lidar_sensor)

    assert out["vehicles"] == []
    assert "static" not in out
