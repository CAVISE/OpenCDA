"""
Unit tests for opencda.core.sensing.perception.perception_manager.

These tests are contract-driven and deterministic:
- No GUI usage (cv2 GUI is disabled via test/conftest.py autouse fixture).
- No real CARLA/Open3D runtime required (stubs are installed via test/conftest.py).
- No infinite loops (we avoid while-wait patterns by pre-populating sensor buffers).

Important: test/conftest.py installs a placeholder module at
sys.modules["opencda.core.sensing.perception.perception_manager"] for legacy tests.
This test module temporarily imports the real production module and restores the placeholder afterwards.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore:A NumPy version .* is required for this version of SciPy.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Unable to import Axes3D.*:UserWarning"),
]


class _O3DPointCloud:
    pass


class _O3DVisualizer:
    def __init__(self):
        self.destroy_window_called = 0

    def destroy_window(self):
        self.destroy_window_called += 1


@pytest.fixture(autouse=True)
def _ensure_open3d_minimal():
    """
    Ensure open3d stub has the minimal surface required by perception_manager constructors.
    """
    o3d = sys.modules["open3d"]
    if not hasattr(o3d, "geometry"):
        o3d.geometry = types.SimpleNamespace()
    if not hasattr(o3d.geometry, "PointCloud"):
        o3d.geometry.PointCloud = lambda: _O3DPointCloud()

    if not hasattr(o3d, "visualization"):
        o3d.visualization = types.SimpleNamespace()
    if not hasattr(o3d.visualization, "Visualizer"):
        o3d.visualization.Visualizer = lambda: _O3DVisualizer()

    return o3d


@pytest.fixture(scope="module")
def perception_manager_module():
    """
    Import the real production module, temporarily replacing the placeholder stub.
    """
    mod_name = "opencda.core.sensing.perception.perception_manager"
    placeholder = sys.modules.get(mod_name)
    sys.modules.pop(mod_name, None)
    try:
        mod = importlib.import_module(mod_name)
        yield mod
    finally:
        sys.modules.pop(mod_name, None)
        if placeholder is not None:
            sys.modules[mod_name] = placeholder


class _FakeBlueprint:
    def __init__(self, blueprint_id: str):
        self.id = blueprint_id
        self.attributes: dict[str, str] = {}
        self.set_calls: list[tuple[str, str]] = []

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value
        self.set_calls.append((key, value))


class _FakeBlueprintLibrary:
    def __init__(self, mapping: dict[str, _FakeBlueprint]):
        self._mapping = mapping

    def find(self, name: str) -> _FakeBlueprint:
        return self._mapping[name]


@dataclass
class _FakeEvent:
    raw_data: object
    frame: int = 1
    timestamp: float = 1.0


class _FakeSensor:
    def __init__(self, *, attributes=None, transform=None):
        self.attributes = attributes if attributes is not None else {}
        self._transform = transform
        self._callback = None
        self.destroy_called = 0

    def listen(self, callback):
        self._callback = callback

    def trigger(self, event: _FakeEvent) -> None:
        if self._callback is None:
            raise AssertionError("Sensor callback is not set (listen was not called).")
        self._callback(event)

    def get_transform(self):
        return self._transform

    def destroy(self):
        self.destroy_called += 1


class _FakeActorList:
    def __init__(self, mapping: dict[str, list]):
        self._mapping = mapping

    def filter(self, pattern: str):
        return list(self._mapping.get(pattern, []))


class _FakeWorld:
    def __init__(self, *, blueprint_library, actors=None, carla_map=None):
        self._blueprint_library = blueprint_library
        self._actors = actors if actors is not None else _FakeActorList({})
        self._map = carla_map if carla_map is not None else Mock()

        self.spawn_calls: list[dict] = []
        self._spawn_plan: dict[str, _FakeSensor] = {}

    def set_spawn_plan(self, blueprint_id: str, sensor: _FakeSensor) -> None:
        self._spawn_plan[blueprint_id] = sensor

    def get_blueprint_library(self):
        return self._blueprint_library

    def spawn_actor(self, blueprint, transform, attach_to=None):
        self.spawn_calls.append({"blueprint": blueprint, "transform": transform, "attach_to": attach_to})

        sensor = self._spawn_plan.get(blueprint.id)
        if sensor is None:
            sensor = _FakeSensor(attributes={}, transform=transform)
        return sensor

    def get_actors(self):
        return self._actors

    def get_map(self):
        return self._map


def _lidar_config() -> dict:
    return {
        "upper_fov": 10.0,
        "lower_fov": -30.0,
        "channels": 32,
        "range": 70.0,
        "points_per_second": 56000,
        "rotation_frequency": 10.0,
        "dropoff_general_rate": 0.0,
        "dropoff_intensity_limit": 1.0,
        "dropoff_zero_intensity": 0.0,
        "noise_stddev": 0.0,
    }


def _make_perception_world_with_camera_and_lidar(*, carla_map=None, actors=None):
    import carla

    cam_bp = _FakeBlueprint("sensor.camera.rgb")
    lidar_bp = _FakeBlueprint("sensor.lidar.ray_cast")
    sem_bp = _FakeBlueprint("sensor.lidar.ray_cast_semantic")

    bl = _FakeBlueprintLibrary(
        {
            "sensor.camera.rgb": cam_bp,
            "sensor.lidar.ray_cast": lidar_bp,
            "sensor.lidar.ray_cast_semantic": sem_bp,
        }
    )

    world = _FakeWorld(blueprint_library=bl, carla_map=carla_map if carla_map is not None else Mock(), actors=actors)
    world.set_spawn_plan(
        "sensor.camera.rgb", _FakeSensor(attributes={"image_size_x": "2", "image_size_y": "2"}, transform=carla.Transform(carla.Location()))
    )
    world.set_spawn_plan("sensor.lidar.ray_cast", _FakeSensor(attributes={}, transform=carla.Transform(carla.Location())))
    world.set_spawn_plan("sensor.lidar.ray_cast_semantic", _FakeSensor(attributes={}, transform=carla.Transform(carla.Location())))
    return world


def _perception_config(*, activate: bool, camera_visualize: int = 0, lidar_visualize: bool = False) -> dict:
    return {
        "activate": activate,
        "camera": {"visualize": camera_visualize, "num": 1, "positions": [(0.0, 0.0, 0.0, 0.0)]},
        "lidar": {"visualize": lidar_visualize, **_lidar_config()},
        "traffic_light_thresh": 50,
    }


def _make_box_corners_centered(center_xyz: tuple[float, float, float], extent_xyz: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center_xyz
    ex, ey, ez = extent_xyz
    x0, x1 = cx - ex, cx + ex
    y0, y1 = cy - ey, cy + ey
    z0, z1 = cz - ez, cz + ez
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


def test_camera_sensor_spawn_point_estimation_without_global_position(perception_manager_module):
    import carla

    CameraSensor = perception_manager_module.CameraSensor
    spawn = CameraSensor.spawn_point_estimation(relative_position=(1.0, 2.0, 3.0, 45.0), global_position=None)

    assert spawn.location == carla.Location(x=1.0, y=2.0, z=3.0)
    assert spawn.rotation == carla.Rotation(roll=0, yaw=45.0, pitch=0)


def test_camera_sensor_spawn_point_estimation_with_global_position(perception_manager_module):
    import carla

    CameraSensor = perception_manager_module.CameraSensor
    spawn = CameraSensor.spawn_point_estimation(relative_position=(1.0, 2.0, 3.0, 90.0), global_position=[10.0, 20.0, 30.0])

    assert spawn.location == carla.Location(x=11.0, y=22.0, z=33.0)
    assert spawn.rotation == carla.Rotation(roll=0, yaw=90.0, pitch=-35)


def test_camera_sensor_init_sets_fov_and_spawns_attached_when_vehicle_provided(perception_manager_module):
    import carla

    CameraSensor = perception_manager_module.CameraSensor

    cam_bp = _FakeBlueprint("sensor.camera.rgb")
    bl = _FakeBlueprintLibrary({"sensor.camera.rgb": cam_bp})

    world = _FakeWorld(blueprint_library=bl, carla_map=Mock())
    sensor = _FakeSensor(attributes={"image_size_x": "2", "image_size_y": "2"}, transform=carla.Transform(carla.Location()))
    world.set_spawn_plan("sensor.camera.rgb", sensor)

    vehicle = Mock(spec_set=["get_world"])
    vehicle.get_world.return_value = world

    CameraSensor(vehicle=vehicle, world=None, relative_position=(0.0, 0.0, 0.0, 0.0), global_position=None)

    assert ("fov", "100") in cam_bp.set_calls
    assert world.spawn_calls
    assert world.spawn_calls[-1]["attach_to"] is vehicle


def test_camera_sensor_rgb_callback_parses_rgba_and_strips_alpha(perception_manager_module):
    import carla

    CameraSensor = perception_manager_module.CameraSensor

    cam_bp = _FakeBlueprint("sensor.camera.rgb")
    bl = _FakeBlueprintLibrary({"sensor.camera.rgb": cam_bp})

    sensor = _FakeSensor(
        attributes={"image_size_x": "2", "image_size_y": "2"},
        transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
    )

    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.camera.rgb", sensor)

    cam = CameraSensor(vehicle=None, world=world, relative_position=(0.0, 0.0, 0.0, 0.0), global_position=None)
    assert sensor._callback is not None

    rgba = np.array(
        [
            10,
            20,
            30,
            255,
            40,
            50,
            60,
            128,
            70,
            80,
            90,
            0,
            1,
            2,
            3,
            4,
        ],
        dtype=np.uint8,
    )

    sensor.trigger(_FakeEvent(raw_data=rgba, frame=7, timestamp=9.5))

    assert cam.frame == 7
    assert cam.timestamp == 9.5
    assert cam.image.shape == (2, 2, 3)
    assert cam.image.dtype == np.uint8

    assert (cam.image[0, 0] == np.array([10, 20, 30], dtype=np.uint8)).all()
    assert (cam.image[0, 1] == np.array([40, 50, 60], dtype=np.uint8)).all()
    assert (cam.image[1, 0] == np.array([70, 80, 90], dtype=np.uint8)).all()


def test_camera_sensor_rgb_callback_bytes_input_raises_current_behavior(perception_manager_module):
    """
    Negative contract: CameraSensor._on_rgb_image_event uses np.array(event.raw_data),
    which does not support bytes input as expected and fails reshape.
    """
    import carla

    CameraSensor = perception_manager_module.CameraSensor

    cam_bp = _FakeBlueprint("sensor.camera.rgb")
    bl = _FakeBlueprintLibrary({"sensor.camera.rgb": cam_bp})

    sensor = _FakeSensor(
        attributes={"image_size_x": "2", "image_size_y": "2"},
        transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
    )

    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.camera.rgb", sensor)

    cam = CameraSensor(vehicle=None, world=world, relative_position=(0.0, 0.0, 0.0, 0.0), global_position=None)
    assert cam.sensor is sensor

    raw = bytes([0] * (2 * 2 * 4))
    with pytest.raises(ValueError):
        sensor.trigger(_FakeEvent(raw_data=raw, frame=1, timestamp=1.0))

    assert cam.image is None


def test_camera_sensor_rgb_callback_bytearray_input_parses_successfully(perception_manager_module):
    """
    Positive contract: a bytearray is an iterable of ints, so np.array(bytearray)
    produces a 1D numeric array and reshape should succeed.
    """
    import carla

    CameraSensor = perception_manager_module.CameraSensor

    cam_bp = _FakeBlueprint("sensor.camera.rgb")
    bl = _FakeBlueprintLibrary({"sensor.camera.rgb": cam_bp})

    sensor = _FakeSensor(
        attributes={"image_size_x": "2", "image_size_y": "2"},
        transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
    )

    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.camera.rgb", sensor)

    cam = CameraSensor(vehicle=None, world=world, relative_position=(0.0, 0.0, 0.0, 0.0), global_position=None)
    assert cam.sensor is sensor

    raw = bytearray(
        [
            10,
            20,
            30,
            255,
            40,
            50,
            60,
            128,
            70,
            80,
            90,
            0,
            1,
            2,
            3,
            4,
        ]
    )
    sensor.trigger(_FakeEvent(raw_data=raw, frame=2, timestamp=3.0))

    assert cam.frame == 2
    assert cam.timestamp == 3.0
    assert cam.image.shape == (2, 2, 3)
    assert (cam.image[0, 0] == np.array([10, 20, 30], dtype=np.uint8)).all()


def test_lidar_sensor_init_sets_blueprint_attributes_and_spawn_point_default(perception_manager_module):
    import carla

    LidarSensor = perception_manager_module.LidarSensor

    lidar_bp = _FakeBlueprint("sensor.lidar.ray_cast")
    bl = _FakeBlueprintLibrary({"sensor.lidar.ray_cast": lidar_bp})

    sensor = _FakeSensor(attributes={}, transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)))
    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.lidar.ray_cast", sensor)

    cfg = _lidar_config()
    lidar = LidarSensor(vehicle=None, world=world, config_yaml=cfg, global_position=None)

    set_dict = dict(lidar_bp.set_calls)
    for k, v in cfg.items():
        assert set_dict[k] == str(v)

    assert world.spawn_calls
    spawn_transform = world.spawn_calls[-1]["transform"]
    assert spawn_transform.location.x == pytest.approx(-0.5)
    assert spawn_transform.location.z == pytest.approx(1.9)

    assert lidar.sensor is sensor
    assert sensor._callback is not None


def test_lidar_sensor_init_attaches_to_vehicle_when_vehicle_provided(perception_manager_module):
    import carla

    LidarSensor = perception_manager_module.LidarSensor

    lidar_bp = _FakeBlueprint("sensor.lidar.ray_cast")
    bl = _FakeBlueprintLibrary({"sensor.lidar.ray_cast": lidar_bp})

    sensor = _FakeSensor(attributes={}, transform=carla.Transform(carla.Location()))
    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.lidar.ray_cast", sensor)

    vehicle = Mock(spec_set=["get_world"])
    vehicle.get_world.return_value = world

    cfg = _lidar_config()
    LidarSensor(vehicle=vehicle, world=None, config_yaml=cfg, global_position=None)

    assert world.spawn_calls
    assert world.spawn_calls[-1]["attach_to"] is vehicle


def test_lidar_sensor_init_uses_global_position(perception_manager_module):
    import carla

    LidarSensor = perception_manager_module.LidarSensor

    lidar_bp = _FakeBlueprint("sensor.lidar.ray_cast")
    bl = _FakeBlueprintLibrary({"sensor.lidar.ray_cast": lidar_bp})

    sensor = _FakeSensor(attributes={}, transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)))
    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.lidar.ray_cast", sensor)

    cfg = _lidar_config()
    LidarSensor(vehicle=None, world=world, config_yaml=cfg, global_position=[1.0, 2.0, 3.0])

    spawn_transform = world.spawn_calls[-1]["transform"]
    assert spawn_transform.location == carla.Location(x=1.0, y=2.0, z=3.0)


def test_lidar_sensor_on_data_event_reshapes_to_n_by_4(perception_manager_module):
    LidarSensor = perception_manager_module.LidarSensor

    dummy = types.SimpleNamespace(data=None, frame=0, timestamp=None)

    pts = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    ev = _FakeEvent(raw_data=pts.tobytes(), frame=11, timestamp=12.5)

    LidarSensor._on_data_event(lambda: dummy, ev)

    assert dummy.frame == 11
    assert dummy.timestamp == 12.5
    assert isinstance(dummy.data, np.ndarray)
    assert dummy.data.shape == (2, 4)
    assert dummy.data.dtype == np.float32
    assert dummy.data == pytest.approx(pts)


def test_semantic_lidar_sensor_on_data_event_parses_structured_buffer(perception_manager_module):
    SemanticLidarSensor = perception_manager_module.SemanticLidarSensor

    dummy = types.SimpleNamespace(points=None, obj_idx=None, obj_tag=None, data=None, frame=0, timestamp=None)

    dt = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("CosAngle", np.float32),
            ("ObjIdx", np.uint32),
            ("ObjTag", np.uint32),
        ]
    )
    raw = np.array(
        [
            (1.0, 2.0, 3.0, 0.0, 10, 14),
            (4.0, 5.0, 6.0, 0.0, 11, 7),
        ],
        dtype=dt,
    ).tobytes()

    ev = _FakeEvent(raw_data=raw, frame=3, timestamp=4.0)
    SemanticLidarSensor._on_data_event(lambda: dummy, ev)

    assert dummy.frame == 3
    assert dummy.timestamp == 4.0
    assert dummy.points.shape == (2, 3)
    assert dummy.obj_idx.tolist() == [10, 11]
    assert dummy.obj_tag.tolist() == [14, 7]


def test_semantic_lidar_sensor_init_sets_blueprint_attributes(perception_manager_module):
    import carla

    SemanticLidarSensor = perception_manager_module.SemanticLidarSensor

    sem_bp = _FakeBlueprint("sensor.lidar.ray_cast_semantic")
    bl = _FakeBlueprintLibrary({"sensor.lidar.ray_cast_semantic": sem_bp})

    sensor = _FakeSensor(attributes={}, transform=carla.Transform(carla.Location()))
    world = _FakeWorld(blueprint_library=bl)
    world.set_spawn_plan("sensor.lidar.ray_cast_semantic", sensor)

    cfg = _lidar_config()
    SemanticLidarSensor(vehicle=None, world=world, config_yaml=cfg, global_position=None)

    set_dict = dict(sem_bp.set_calls)
    for k in ("upper_fov", "lower_fov", "channels", "range", "points_per_second", "rotation_frequency"):
        assert set_dict[k] == str(cfg[k])


def test_perception_manager_init_exits_when_activate_and_no_ml_manager(perception_manager_module):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=True)
    cav_world = Mock()
    cav_world.ml_manager = None

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    with pytest.raises(SystemExit):
        PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)


def test_perception_manager_init_spawns_camera_and_lidar_when_activate_true(perception_manager_module):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=True, camera_visualize=0, lidar_visualize=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    world = _make_perception_world_with_camera_and_lidar()
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    assert pm.activate is True
    assert pm.rgb_camera is not None
    assert len(pm.rgb_camera) == 1
    assert pm.lidar is not None


def test_perception_manager_init_spawns_semantic_lidar_when_data_dump_true(perception_manager_module):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False, camera_visualize=0, lidar_visualize=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    world = _make_perception_world_with_camera_and_lidar()
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=True, carla_world=world)

    assert pm.data_dump is True
    assert pm.semantic_lidar is not None


def test_perception_manager_init_spawns_camera_when_camera_visualize_enabled(perception_manager_module):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False, camera_visualize=1, lidar_visualize=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    world = _make_perception_world_with_camera_and_lidar()
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    assert pm.rgb_camera is not None
    assert len(pm.rgb_camera) == 1
    assert pm.rgb_camera[0].image is None


def test_perception_manager_init_spawns_lidar_and_visualizer_when_lidar_visualize_enabled(perception_manager_module, monkeypatch):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False, camera_visualize=0, lidar_visualize=True)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    world = _make_perception_world_with_camera_and_lidar()
    vis = _O3DVisualizer()
    monkeypatch.setattr(perception_manager_module, "o3d_visualizer_init", lambda _actor_id: vis)

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=7, data_dump=False, carla_world=world)

    assert pm.lidar is not None
    assert pm.o3d_vis is vis


def test_perception_manager_vehicle_mode_sets_carla_id_and_attaches_sensors(perception_manager_module):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False, camera_visualize=1, lidar_visualize=True)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    world = _make_perception_world_with_camera_and_lidar()
    vehicle = Mock(spec_set=["id", "get_world"])
    vehicle.id = 123
    vehicle.get_world.return_value = world

    pm = PerceptionManager(vehicle=vehicle, config_yaml=cfg, cav_world=cav_world, infra_id=999, data_dump=False, carla_world=None)
    assert pm.carla_id == 123

    assert world.spawn_calls
    assert any(call["attach_to"] is vehicle for call in world.spawn_calls)


def test_detect_dispatches_between_activate_and_deactivate_modes(perception_manager_module):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    pm.deactivate_mode = Mock(side_effect=lambda objects: {**objects, "vehicles": ["x"]})
    ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    out = pm.detect(ego_pos)
    assert out["vehicles"] == ["x"]
    assert pm.count == 1
    assert pm.ego_pos is ego_pos
    pm.deactivate_mode.assert_called_once()

    pm.activate = True
    pm.activate_mode = Mock(side_effect=lambda objects: {**objects, "vehicles": ["y"]})
    out2 = pm.detect(ego_pos)
    assert out2["vehicles"] == ["y"]
    assert pm.count == 2
    pm.activate_mode.assert_called_once()


def test_dist_uses_actor_location_distance(perception_manager_module):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    pm.ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())
    actor = Mock(spec_set=["get_location"])
    actor.get_location.return_value = carla.Location(x=3.0, y=4.0, z=0.0)
    assert pm.dist(actor) == pytest.approx(5.0)


def test_deactivate_mode_filters_by_distance_and_excludes_self(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    class _Veh:
        def __init__(self, actor_id: int, location: carla.Location):
            self.id = actor_id
            self._location = location

        def get_location(self):
            return self._location

    ego = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    v_close = _Veh(actor_id=2, location=carla.Location(x=10.0, y=0.0, z=0.0))
    v_far = _Veh(actor_id=3, location=carla.Location(x=200.0, y=0.0, z=0.0))
    v_self = _Veh(actor_id=1, location=carla.Location(x=5.0, y=0.0, z=0.0))

    actors = _FakeActorList({"*vehicle*": [v_close, v_far, v_self]})
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), actors=actors, carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = ego

    constructed = []

    class _OV:
        def __init__(self, _c1, _c2, vehicle, lidar, sumo2carla_ids):  # noqa: ARG002
            constructed.append({"vehicle": vehicle, "lidar": lidar, "sumo2carla_ids": sumo2carla_ids})

    monkeypatch.setattr(perception_manager_module, "ObstacleVehicle", _OV)

    out = pm.deactivate_mode({"vehicles": [], "traffic_lights": []})

    assert "vehicles" in out
    assert len(out["vehicles"]) == 1
    assert constructed[0]["vehicle"] is v_close


def test_deactivate_mode_calls_filter_vehicle_out_sensor_when_data_dump_true(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    class _Veh:
        def __init__(self, actor_id: int, location: carla.Location):
            self.id = actor_id
            self._location = location

        def get_location(self):
            return self._location

    ego = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())
    v_close = _Veh(actor_id=2, location=carla.Location(x=10.0, y=0.0, z=0.0))
    actors = _FakeActorList({"*vehicle*": [v_close]})
    world = _make_perception_world_with_camera_and_lidar(actors=actors)

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=True, carla_world=world)
    pm.ego_pos = ego

    called = {"n": 0}
    monkeypatch.setattr(pm, "filter_vehicle_out_sensor", lambda vehicle_list: (called.__setitem__("n", called["n"] + 1) or vehicle_list))
    monkeypatch.setattr(perception_manager_module, "ObstacleVehicle", lambda *_args, **_kwargs: "ov")

    out = pm.deactivate_mode({"vehicles": [], "traffic_lights": []})
    assert called["n"] == 1
    assert out["vehicles"] == ["ov"]


def test_filter_vehicle_out_sensor_contracts(perception_manager_module):
    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    class _Veh:
        def __init__(self, actor_id: int):
            self.id = actor_id

    vehicles = [_Veh(1), _Veh(2), _Veh(3)]

    pm.semantic_lidar = types.SimpleNamespace(obj_idx=None, obj_tag=None)
    assert pm.filter_vehicle_out_sensor(vehicles) == vehicles

    pm.semantic_lidar = types.SimpleNamespace(obj_idx=np.array([1, 2, 3], dtype=np.uint32), obj_tag=np.array([14, 14], dtype=np.uint32))
    out = pm.filter_vehicle_out_sensor(vehicles)
    assert [v.id for v in out] == [1, 2]

    pm.semantic_lidar = types.SimpleNamespace(
        obj_idx=np.array([1, 99, 2, 2], dtype=np.uint32),
        obj_tag=np.array([14, 7, 14, 14], dtype=np.uint32),
    )
    out2 = pm.filter_vehicle_out_sensor(vehicles)
    assert [v.id for v in out2] == [1, 2]


def test_speed_retrieve_matches_by_xy_threshold_and_sets_velocity_and_id(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    class _Actor:
        def __init__(self, actor_id: int, location: carla.Location, velocity: carla.Vector3D):
            self.id = actor_id
            self._location = location
            self._velocity = velocity

        def get_location(self):
            return self._location

        def get_velocity(self):
            return self._velocity

    ego = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    actor = _Actor(10, carla.Location(x=20.0, y=0.0, z=0.0), carla.Vector3D(x=3.0, y=0.0, z=0.0))
    actors = _FakeActorList({"*vehicle*": [actor]})
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), actors=actors, carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = ego

    monkeypatch.setattr(
        perception_manager_module, "get_speed", lambda ov: abs(ov.get_velocity().x) + abs(ov.get_velocity().y) + abs(ov.get_velocity().z)
    )

    corners = np.array(
        [
            [20.2, 0.1, 0.0],
            [20.2, 0.1, 1.0],
            [20.2, 0.9, 0.0],
            [20.2, 0.9, 1.0],
            [21.0, 0.1, 0.0],
            [21.0, 0.1, 1.0],
            [21.0, 0.9, 0.0],
            [21.0, 0.9, 1.0],
        ],
        dtype=float,
    )
    ov = ObstacleVehicle(corners, o3d_bbx=object())

    objects = {"vehicles": [ov]}
    pm.speed_retrieve(objects)

    assert ov.carla_id == 10
    assert ov.get_velocity() == carla.Vector3D(x=3.0, y=0.0, z=0.0)


def test_speed_retrieve_sumo_override(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {"sumo": 10}

    class _Actor:
        def __init__(self, actor_id: int, location: carla.Location, velocity: carla.Vector3D):
            self.id = actor_id
            self._location = location
            self._velocity = velocity

        def get_location(self):
            return self._location

        def get_velocity(self):
            return self._velocity

    ego = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    actor = _Actor(10, carla.Location(x=20.0, y=0.0, z=0.0), carla.Vector3D(x=0.5, y=0.0, z=0.0))
    actors = _FakeActorList({"*vehicle*": [actor]})
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), actors=actors, carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = ego

    monkeypatch.setattr(perception_manager_module, "get_speed", lambda ov: abs(ov.get_velocity().x))
    monkeypatch.setattr(perception_manager_module, "get_speed_sumo", lambda _mapping, _carla_id: 12.0)

    corners = np.array(
        [
            [20.0, 0.0, 0.0],
            [20.0, 0.0, 1.0],
            [20.0, 1.0, 0.0],
            [20.0, 1.0, 1.0],
            [21.0, 0.0, 0.0],
            [21.0, 0.0, 1.0],
            [21.0, 1.0, 0.0],
            [21.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    ov = ObstacleVehicle(corners, o3d_bbx=object())
    pm.speed_retrieve({"vehicles": [ov]})

    assert ov.carla_id == 10
    assert ov.get_velocity() == carla.Vector3D(x=12.0, y=0.0, z=0.0)


def test_speed_retrieve_does_not_override_already_matched_obstacle(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    class _Actor:
        def __init__(self, actor_id: int, location: carla.Location, velocity: carla.Vector3D):
            self.id = actor_id
            self._location = location
            self._velocity = velocity

        def get_location(self):
            return self._location

        def get_velocity(self):
            return self._velocity

    actor = _Actor(10, carla.Location(x=20.0, y=0.0, z=0.0), carla.Vector3D(x=3.0, y=0.0, z=0.0))
    actors = _FakeActorList({"*vehicle*": [actor]})
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), actors=actors, carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    monkeypatch.setattr(perception_manager_module, "get_speed", lambda _ov: 1.0)

    corners = _make_box_corners_centered(center_xyz=(20.0, 0.0, 0.5), extent_xyz=(0.5, 0.5, 0.5))
    ov = ObstacleVehicle(corners, o3d_bbx=object())

    orig_vel = ov.get_velocity()
    pm.speed_retrieve({"vehicles": [ov]})

    assert ov.carla_id == -1
    assert ov.get_velocity() == orig_vel


def test_speed_retrieve_boundary_thresholds_match_and_non_match(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    class _Actor:
        def __init__(self, actor_id: int, location: carla.Location, velocity: carla.Vector3D):
            self.id = actor_id
            self._location = location
            self._velocity = velocity

        def get_location(self):
            return self._location

        def get_velocity(self):
            return self._velocity

    actor_loc = carla.Location(x=20.0, y=10.0, z=0.0)
    actor = _Actor(10, actor_loc, carla.Vector3D(x=3.0, y=0.0, z=0.0))
    actors = _FakeActorList({"*vehicle*": [actor]})
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), actors=actors, carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    monkeypatch.setattr(perception_manager_module, "get_speed", lambda _ov: 0.0)

    corners_match = _make_box_corners_centered(center_xyz=(actor_loc.x + 3.0, actor_loc.y + 3.0, 0.5), extent_xyz=(0.5, 0.5, 0.5))
    corners_no_match = _make_box_corners_centered(center_xyz=(actor_loc.x + 3.0001, actor_loc.y, 0.5), extent_xyz=(0.5, 0.5, 0.5))

    ov_match = ObstacleVehicle(corners_match, o3d_bbx=object())
    ov_no_match = ObstacleVehicle(corners_no_match, o3d_bbx=object())

    pm.speed_retrieve({"vehicles": [ov_match, ov_no_match]})

    assert ov_match.carla_id == 10
    assert ov_match.get_velocity() == carla.Vector3D(x=3.0, y=0.0, z=0.0)

    assert ov_no_match.carla_id == -1


class _FakeForwardVec:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class _FakeWpTransform:
    def __init__(self, location, forward_vec):
        self.location = location
        self._fv = forward_vec

    def get_forward_vector(self):
        return self._fv


class _FakeWaypoint:
    def __init__(self, *, road_id: int, location, forward_vec, is_intersection: bool, next_wp=None):
        self.road_id = road_id
        self.transform = _FakeWpTransform(location, forward_vec)
        self.is_intersection = is_intersection
        self._next = next_wp

    def next(self, _dist: float):
        return [self._next] if self._next is not None else [None]


class _FakeMap:
    def __init__(self, by_key: dict[tuple[float, float, float], _FakeWaypoint]):
        self._by_key = by_key

    @staticmethod
    def _key(loc) -> tuple[float, float, float]:
        return (float(loc.x), float(loc.y), float(loc.z))

    def get_waypoint(self, location):
        return self._by_key[self._key(location)]


def test_get_active_light_selects_light_on_same_road_and_forward_direction(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    TrafficLight = perception_manager_module.TrafficLight

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    vehicle_loc = carla.Location(x=0.0, y=0.0, z=0.0)
    trigger_loc = carla.Location(x=10.0, y=0.0, z=0.0)
    intersection_loc = carla.Location(x=11.0, y=0.0, z=0.0)

    fv = _FakeForwardVec(1.0, 0.0, 0.0)

    veh_wp = _FakeWaypoint(road_id=7, location=vehicle_loc, forward_vec=fv, is_intersection=False)
    tl_wp = _FakeWaypoint(
        road_id=7,
        location=trigger_loc,
        forward_vec=fv,
        is_intersection=False,
        next_wp=_FakeWaypoint(road_id=7, location=intersection_loc, forward_vec=fv, is_intersection=True),
    )

    fake_map = _FakeMap(
        {
            (vehicle_loc.x, vehicle_loc.y, vehicle_loc.z): veh_wp,
            (trigger_loc.x, trigger_loc.y, trigger_loc.z): tl_wp,
        }
    )

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=fake_map)
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    tl_actor = Mock(spec_set=[])
    monkeypatch.setattr(TrafficLight, "get_trafficlight_trigger_location", staticmethod(lambda _tl: trigger_loc))

    active, location = pm._get_active_light([tl_actor], vehicle_loc, veh_wp)
    assert active is tl_actor
    assert location == trigger_loc


def test_get_active_light_returns_last_non_intersection_before_intersection(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    TrafficLight = perception_manager_module.TrafficLight

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    vehicle_loc = carla.Location(x=0.0, y=0.0, z=0.0)
    trigger_loc = carla.Location(x=10.0, y=0.0, z=0.0)
    pre_int_loc = carla.Location(x=10.5, y=0.0, z=0.0)
    intersection_loc = carla.Location(x=11.0, y=0.0, z=0.0)

    fv = _FakeForwardVec(1.0, 0.0, 0.0)
    veh_wp = _FakeWaypoint(road_id=7, location=vehicle_loc, forward_vec=fv, is_intersection=False)

    wp_intersection = _FakeWaypoint(road_id=7, location=intersection_loc, forward_vec=fv, is_intersection=True)
    wp_pre = _FakeWaypoint(road_id=7, location=pre_int_loc, forward_vec=fv, is_intersection=False, next_wp=wp_intersection)
    tl_wp = _FakeWaypoint(road_id=7, location=trigger_loc, forward_vec=fv, is_intersection=False, next_wp=wp_pre)

    fake_map = _FakeMap(
        {
            (vehicle_loc.x, vehicle_loc.y, vehicle_loc.z): veh_wp,
            (trigger_loc.x, trigger_loc.y, trigger_loc.z): tl_wp,
            (pre_int_loc.x, pre_int_loc.y, pre_int_loc.z): wp_pre,
            (intersection_loc.x, intersection_loc.y, intersection_loc.z): wp_intersection,
        }
    )

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=fake_map)
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    tl_actor = Mock(spec_set=[])
    monkeypatch.setattr(TrafficLight, "get_trafficlight_trigger_location", staticmethod(lambda _tl: trigger_loc))

    active, location = pm._get_active_light([tl_actor], vehicle_loc, veh_wp)
    assert active is tl_actor
    assert location == pre_int_loc


def test_get_active_light_returns_none_on_wrong_road_or_reverse_direction(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    TrafficLight = perception_manager_module.TrafficLight

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    vehicle_loc = carla.Location(x=0.0, y=0.0, z=0.0)
    trigger_loc = carla.Location(x=10.0, y=0.0, z=0.0)

    fv_forward = _FakeForwardVec(1.0, 0.0, 0.0)
    fv_reverse = _FakeForwardVec(-1.0, 0.0, 0.0)

    veh_wp = _FakeWaypoint(road_id=7, location=vehicle_loc, forward_vec=fv_forward, is_intersection=False)
    tl_wp_wrong_road = _FakeWaypoint(road_id=8, location=trigger_loc, forward_vec=fv_forward, is_intersection=False)
    tl_wp_reverse = _FakeWaypoint(road_id=7, location=trigger_loc, forward_vec=fv_reverse, is_intersection=False)

    tl_actor = Mock(spec_set=[])

    world1 = _FakeWorld(
        blueprint_library=_FakeBlueprintLibrary({}),
        carla_map=_FakeMap({(vehicle_loc.x, vehicle_loc.y, vehicle_loc.z): veh_wp, (trigger_loc.x, trigger_loc.y, trigger_loc.z): tl_wp_wrong_road}),
    )
    pm1 = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world1)
    monkeypatch.setattr(TrafficLight, "get_trafficlight_trigger_location", staticmethod(lambda _tl: trigger_loc))
    assert pm1._get_active_light([tl_actor], vehicle_loc, veh_wp) == (None, None)

    world2 = _FakeWorld(
        blueprint_library=_FakeBlueprintLibrary({}),
        carla_map=_FakeMap({(vehicle_loc.x, vehicle_loc.y, vehicle_loc.z): veh_wp, (trigger_loc.x, trigger_loc.y, trigger_loc.z): tl_wp_reverse}),
    )
    pm2 = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world2)
    monkeypatch.setattr(TrafficLight, "get_trafficlight_trigger_location", staticmethod(lambda _tl: trigger_loc))
    assert pm2._get_active_light([tl_actor], vehicle_loc, veh_wp) == (None, None)


def test_get_active_light_next_none_breaks_and_returns_current_waypoint_location(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager
    TrafficLight = perception_manager_module.TrafficLight

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    vehicle_loc = carla.Location(x=0.0, y=0.0, z=0.0)
    trigger_loc = carla.Location(x=10.0, y=0.0, z=0.0)

    fv = _FakeForwardVec(1.0, 0.0, 0.0)
    veh_wp = _FakeWaypoint(road_id=7, location=vehicle_loc, forward_vec=fv, is_intersection=False)
    tl_wp = _FakeWaypoint(road_id=7, location=trigger_loc, forward_vec=fv, is_intersection=False, next_wp=None)

    fake_map = _FakeMap(
        {
            (vehicle_loc.x, vehicle_loc.y, vehicle_loc.z): veh_wp,
            (trigger_loc.x, trigger_loc.y, trigger_loc.z): tl_wp,
        }
    )

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=fake_map)
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    tl_actor = Mock(spec_set=[])
    monkeypatch.setattr(TrafficLight, "get_trafficlight_trigger_location", staticmethod(lambda _tl: trigger_loc))

    active, location = pm._get_active_light([tl_actor], vehicle_loc, veh_wp)
    assert active is tl_actor
    assert location == trigger_loc


def test_retrieve_traffic_lights_adds_active_light(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    world.get_actors = lambda: _FakeActorList({"traffic.traffic_light*": [Mock(spec_set=[])]})

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    active_tl = Mock(spec_set=["get_state"])
    active_tl.get_state.return_value = "GREEN"
    trigger_loc = carla.Location(x=1.0, y=2.0, z=3.0)

    monkeypatch.setattr(pm, "_get_active_light", lambda _tl_list, _vl, _vw: (active_tl, trigger_loc))
    monkeypatch.setattr(
        pm._map,
        "get_waypoint",
        lambda _loc: Mock(road_id=1, transform=Mock(get_forward_vector=lambda: Mock(x=1, y=0, z=0)), is_intersection=True, next=lambda _d: [None]),
    )

    out = pm.retrieve_traffic_lights({"vehicles": [], "traffic_lights": []})
    assert "traffic_lights" in out
    assert len(out["traffic_lights"]) == 1
    tl = out["traffic_lights"][0]
    assert tl.actor is active_tl
    assert tl.get_location() == trigger_loc
    assert tl.get_state() == "GREEN"


def test_retrieve_traffic_lights_returns_empty_when_no_active_light(perception_manager_module, monkeypatch):
    import carla

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    world.get_actors = lambda: _FakeActorList({"traffic.traffic_light*": [Mock(spec_set=[])]})

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)
    pm.ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    monkeypatch.setattr(pm, "_get_active_light", lambda _tl_list, _vl, _vw: (None, None))
    monkeypatch.setattr(
        pm._map, "get_waypoint", lambda _loc: Mock(road_id=1, transform=Mock(get_forward_vector=lambda: Mock(x=1, y=0, z=0)), is_intersection=True)
    )

    out = pm.retrieve_traffic_lights({"vehicles": [], "traffic_lights": []})
    assert out["traffic_lights"] == []


def test_destroy_calls_destroy_on_spawned_sensors_and_visualizers(perception_manager_module, monkeypatch):
    import carla
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cam_bp = _FakeBlueprint("sensor.camera.rgb")
    lidar_bp = _FakeBlueprint("sensor.lidar.ray_cast")
    sem_bp = _FakeBlueprint("sensor.lidar.ray_cast_semantic")

    bl = _FakeBlueprintLibrary(
        {
            "sensor.camera.rgb": cam_bp,
            "sensor.lidar.ray_cast": lidar_bp,
            "sensor.lidar.ray_cast_semantic": sem_bp,
        }
    )

    cam_sensor = _FakeSensor(attributes={"image_size_x": "2", "image_size_y": "2"}, transform=carla.Transform(carla.Location()))
    lidar_sensor = _FakeSensor(attributes={}, transform=carla.Transform(carla.Location()))
    sem_sensor = _FakeSensor(attributes={}, transform=carla.Transform(carla.Location()))

    world = _FakeWorld(blueprint_library=bl, carla_map=Mock())
    world.set_spawn_plan("sensor.camera.rgb", cam_sensor)
    world.set_spawn_plan("sensor.lidar.ray_cast", lidar_sensor)
    world.set_spawn_plan("sensor.lidar.ray_cast_semantic", sem_sensor)

    cfg = _perception_config(activate=False, camera_visualize=1, lidar_visualize=True)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    monkeypatch.setattr(perception_manager_module, "o3d_visualizer_init", lambda _actor_id: _O3DVisualizer())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=True, carla_world=world)

    destroy_all = Mock()
    monkeypatch.setattr(cv2, "destroyAllWindows", destroy_all, raising=True)

    pm.destroy()

    assert cam_sensor.destroy_called == 1
    assert lidar_sensor.destroy_called == 1
    assert sem_sensor.destroy_called == 1
    assert pm.o3d_vis.destroy_window_called == 1
    destroy_all.assert_called_once_with()


def test_visualize_3d_bbx_front_camera_draws_only_objects_in_fov(perception_manager_module, monkeypatch):
    import carla
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    cam_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0))
    cam_sensor = Mock(spec_set=["get_transform"])
    cam_sensor.get_transform.return_value = cam_transform
    pm.rgb_camera = [types.SimpleNamespace(sensor=cam_sensor)]

    class _OV:
        def __init__(self, loc):
            self._loc = loc

        def get_location(self):
            return self._loc

    ov_in = _OV(carla.Location(x=10.0, y=0.0, z=0.0))
    ov_out = _OV(carla.Location(x=-10.0, y=0.0, z=0.0))
    objects = {"vehicles": [ov_in, ov_out]}

    monkeypatch.setattr(perception_manager_module, "cal_distance_angle", lambda _a, _b, _yaw: (0.0, 10.0) if _a.x > 0 else (0.0, 70.0))
    monkeypatch.setattr(perception_manager_module.st, "get_2d_bb", lambda *_args: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))

    rect = Mock(side_effect=lambda img, *_args, **_kwargs: img)
    monkeypatch.setattr(cv2, "rectangle", rect, raising=True)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = pm.visualize_3d_bbx_front_camera(objects, img, camera_index=0)

    assert out is img
    assert rect.call_count == 1
    ((_, pt1, pt2, color, thickness), _) = rect.call_args
    assert pt1 == (1, 2)
    assert pt2 == (3, 4)
    assert color == (255, 0, 0)
    assert thickness == 2


def test_deactivate_mode_camera_visualize_calls_visualize_and_imshow(perception_manager_module, monkeypatch):
    import carla
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False, camera_visualize=1, lidar_visualize=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    cav_world.sumo2carla_ids = {}

    class _Veh:
        def __init__(self, actor_id: int, location: carla.Location):
            self.id = actor_id
            self._location = location

        def get_location(self):
            return self._location

    actors = _FakeActorList({"*vehicle*": [_Veh(10, carla.Location(x=10.0, y=0.0, z=0.0))]})
    world = _make_perception_world_with_camera_and_lidar(actors=actors)

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=5, data_dump=False, carla_world=world)
    pm.ego_pos = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation())

    pm.rgb_camera[0].image = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(perception_manager_module, "ObstacleVehicle", lambda *_args, **_kwargs: Mock(spec_set=["get_location"]))
    vis = Mock(side_effect=lambda objects, image, _idx: image)
    monkeypatch.setattr(pm, "visualize_3d_bbx_front_camera", vis)

    monkeypatch.setattr(cv2, "resize", lambda img, *_args, **_kwargs: img, raising=True)
    show = Mock()
    wait = Mock(return_value=1)
    monkeypatch.setattr(cv2, "imshow", show, raising=True)
    monkeypatch.setattr(cv2, "waitKey", wait, raising=True)

    out = pm.deactivate_mode({"vehicles": [], "traffic_lights": []})
    assert "vehicles" in out

    assert vis.call_count == 1
    assert show.call_count == 1
    assert wait.call_count == 1


def test_activate_mode_contract_without_visualization(perception_manager_module, monkeypatch):
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()

    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    pm.ml_manager = Mock()
    pm.camera_num = 1
    pm.camera_visualize = 0
    pm.lidar_visualize = False

    pm.rgb_camera = [types.SimpleNamespace(image=np.zeros((2, 2, 3), dtype=np.uint8), sensor=object())]
    pm.lidar = types.SimpleNamespace(sensor=object(), data=np.zeros((1, 4), dtype=np.float32), o3d_pointcloud=_O3DPointCloud())

    yolo = types.SimpleNamespace(xyxy=[object()])
    pm.ml_manager.object_detector = Mock(return_value=yolo)

    monkeypatch.setattr(cv2, "cvtColor", lambda img, _code: img, raising=True)
    monkeypatch.setattr(
        perception_manager_module.st, "project_lidar_to_camera", lambda *_args: (np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((1, 3), dtype=float))
    )

    fusion_calls = {"n": 0}

    def _fusion(objects, _yolo_bbx, _lidar_3d, _projected, _lidar_sensor):
        fusion_calls["n"] += 1
        objects.setdefault("vehicles", []).append("veh")
        return objects

    monkeypatch.setattr(perception_manager_module, "o3d_camera_lidar_fusion", _fusion)

    pm.speed_retrieve = Mock()
    pm.retrieve_traffic_lights = Mock(side_effect=lambda objects: objects)

    out = pm.activate_mode({"vehicles": [], "traffic_lights": []})

    assert fusion_calls["n"] == 1
    assert out["vehicles"] == ["veh"]
    pm.speed_retrieve.assert_called_once()
    pm.retrieve_traffic_lights.assert_called_once()


def test_activate_mode_calls_fusion_and_speed_retrieve_per_camera(perception_manager_module, monkeypatch):
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = {
        "activate": False,
        "camera": {"visualize": 0, "num": 2, "positions": [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 90.0)]},
        "lidar": {"visualize": False, **_lidar_config()},
        "traffic_light_thresh": 50,
    }
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    pm.ml_manager = Mock()
    pm.camera_num = 2
    pm.camera_visualize = 0
    pm.lidar_visualize = False

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    pm.rgb_camera = [
        types.SimpleNamespace(image=img, sensor=object()),
        types.SimpleNamespace(image=img, sensor=object()),
    ]
    pm.lidar = types.SimpleNamespace(sensor=object(), data=np.zeros((2, 4), dtype=np.float32), o3d_pointcloud=_O3DPointCloud())

    yolo = types.SimpleNamespace(xyxy=[object(), object()])
    pm.ml_manager.object_detector = Mock(return_value=yolo)

    monkeypatch.setattr(cv2, "cvtColor", lambda image, _code: image, raising=True)
    monkeypatch.setattr(perception_manager_module.st, "project_lidar_to_camera", lambda *_args: (img, np.zeros((2, 3), dtype=float)))

    fusion = Mock(side_effect=lambda objects, *_args: objects)
    monkeypatch.setattr(perception_manager_module, "o3d_camera_lidar_fusion", fusion)

    pm.speed_retrieve = Mock()
    pm.retrieve_traffic_lights = Mock(side_effect=lambda objects: objects)

    out = pm.activate_mode({"vehicles": [], "traffic_lights": []})
    assert out["vehicles"] == []
    assert fusion.call_count == 2
    assert pm.speed_retrieve.call_count == 2


def test_activate_mode_camera_visualize_calls_draw_and_imshow(perception_manager_module, monkeypatch):
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())

    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    pm.ml_manager = Mock()
    pm.camera_num = 1
    pm.camera_visualize = 1
    pm.lidar_visualize = False

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    pm.rgb_camera = [types.SimpleNamespace(image=img, sensor=object())]
    pm.lidar = types.SimpleNamespace(sensor=object(), data=np.zeros((1, 4), dtype=np.float32), o3d_pointcloud=_O3DPointCloud())

    yolo = types.SimpleNamespace(xyxy=[object()])
    pm.ml_manager.object_detector = Mock(return_value=yolo)
    pm.ml_manager.draw_2d_box = Mock(side_effect=lambda _det, image, _i: image)

    monkeypatch.setattr(cv2, "cvtColor", lambda image, _code: image, raising=True)
    monkeypatch.setattr(perception_manager_module.st, "project_lidar_to_camera", lambda *_args: (img, np.zeros((1, 3), dtype=float)))
    monkeypatch.setattr(perception_manager_module, "o3d_camera_lidar_fusion", lambda objects, *_args: objects)

    show = Mock()
    wait = Mock(return_value=1)
    monkeypatch.setattr(cv2, "imshow", show, raising=True)
    monkeypatch.setattr(cv2, "waitKey", wait, raising=True)

    pm.speed_retrieve = Mock()
    pm.retrieve_traffic_lights = Mock(side_effect=lambda objects: objects)

    pm.activate_mode({"vehicles": [], "traffic_lights": []})

    pm.ml_manager.draw_2d_box.assert_called_once()
    assert show.call_count == 1
    assert wait.call_count == 1


def test_activate_mode_lidar_visualize_calls_encode_and_show(perception_manager_module, monkeypatch):
    import cv2

    PerceptionManager = perception_manager_module.PerceptionManager

    cfg = _perception_config(activate=False)
    cav_world = Mock()
    cav_world.ml_manager = Mock()
    world = _FakeWorld(blueprint_library=_FakeBlueprintLibrary({}), carla_map=Mock())
    pm = PerceptionManager(vehicle=None, config_yaml=cfg, cav_world=cav_world, infra_id=1, data_dump=False, carla_world=world)

    pm.ml_manager = Mock()
    pm.camera_num = 1
    pm.camera_visualize = 0
    pm.lidar_visualize = True
    pm.o3d_vis = _O3DVisualizer()

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    pm.rgb_camera = [types.SimpleNamespace(image=img, sensor=object())]
    pm.lidar = types.SimpleNamespace(sensor=object(), data=np.zeros((1, 4), dtype=np.float32), o3d_pointcloud=_O3DPointCloud())

    yolo = types.SimpleNamespace(xyxy=[object()])
    pm.ml_manager.object_detector = Mock(return_value=yolo)

    monkeypatch.setattr(cv2, "cvtColor", lambda image, _code: image, raising=True)
    monkeypatch.setattr(perception_manager_module.st, "project_lidar_to_camera", lambda *_args: (img, np.zeros((1, 3), dtype=float)))
    monkeypatch.setattr(perception_manager_module, "o3d_camera_lidar_fusion", lambda objects, *_args: objects)

    encode = Mock()
    show = Mock()
    monkeypatch.setattr(perception_manager_module, "o3d_pointcloud_encode", encode)
    monkeypatch.setattr(perception_manager_module, "o3d_visualizer_show", show)

    pm.speed_retrieve = Mock()
    pm.retrieve_traffic_lights = Mock(side_effect=lambda objects: objects)

    pm.activate_mode({"vehicles": [], "traffic_lights": []})

    encode.assert_called_once()
    show.assert_called_once()
