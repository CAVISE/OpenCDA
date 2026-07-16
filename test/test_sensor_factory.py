"""Tests for preparing and grouping batch-created CARLA sensors."""

from types import SimpleNamespace
from unittest.mock import Mock

import carla

from opencda.core.common.agent import AgentType
from opencda.core.sensing import sensor_factory
from opencda.core.sensing.perception.perception_manager import PerceptionRequirements
from opencda.core.sensing.sensor_factory import build_sensor_actor_bundles, prepare_sensor_spawn_specs
from opencda.core.sensing.sensor_types import AgentSensorContext, SensorType


def _blueprint_library() -> Mock:
    library = Mock()

    def find(type_id):
        blueprint = Mock()
        blueprint.id = type_id
        return blueprint

    library.find.side_effect = find
    return library


def _patch_perception_sensor_adapters(mocker) -> None:
    camera_sensor = Mock()
    camera_sensor.prepare_blueprint.side_effect = lambda library: library.find("sensor.camera.rgb")
    camera_sensor.spawn_point_estimation.side_effect = lambda relative, global_position: carla.Transform(
        carla.Location(
            x=(global_position[0] if global_position is not None else 0.0) + relative[0],
            y=(global_position[1] if global_position is not None else 0.0) + relative[1],
            z=(global_position[2] if global_position is not None else 0.0) + relative[2],
        )
    )
    lidar_sensor = Mock()
    lidar_sensor.prepare_blueprint.side_effect = lambda library, config: library.find("sensor.lidar.ray_cast")
    lidar_sensor.spawn_point_estimation.side_effect = lambda global_position: carla.Transform(
        carla.Location(*(global_position[:3] if global_position is not None else [-0.5, 0.0, 1.9]))
    )
    semantic_lidar_sensor = Mock()
    semantic_lidar_sensor.prepare_blueprint.side_effect = lambda library, config: library.find("sensor.lidar.ray_cast_semantic")
    semantic_lidar_sensor.spawn_point_estimation.side_effect = lidar_sensor.spawn_point_estimation.side_effect

    mocker.patch.object(sensor_factory, "CameraSensor", camera_sensor)
    mocker.patch.object(sensor_factory, "LidarSensor", lidar_sensor)
    mocker.patch.object(sensor_factory, "SemanticLidarSensor", semantic_lidar_sensor)


def _agent_config(*, sensor_localization: bool, camera_num: int = 0, spawn_position=None):
    return {
        "spawn_position": spawn_position or [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        "sensing": {
            "localization": {
                "provider": "sensor" if sensor_localization else "gt",
                "dt": 0.05,
                "gnss": {
                    "noise_alt_stddev": 0.1,
                    "noise_lat_stddev": 0.2,
                    "noise_lon_stddev": 0.3,
                },
            },
            "perception": {
                "activate": False,
                "camera": {
                    "visualize": 0,
                    "num": camera_num,
                    "positions": [[float(index), 0.0, 1.0, 0.0] for index in range(camera_num)],
                },
                "lidar": {
                    "visualize": False,
                    "channels": 32,
                    "range": 50,
                    "points_per_second": 100000,
                    "rotation_frequency": 20,
                    "upper_fov": 10.0,
                    "lower_fov": -30.0,
                    "dropoff_general_rate": 0.0,
                    "dropoff_intensity_limit": 1.0,
                    "dropoff_zero_intensity": 0.0,
                    "noise_stddev": 0.0,
                },
            },
        },
    }


def test_prepare_cav_sensor_specs_attaches_localization_and_collision_sensors(mocker):
    _patch_perception_sensor_adapters(mocker)
    actor = SimpleNamespace(id=10)
    context = AgentSensorContext(
        agent_index=0,
        agent_type=AgentType.CAV,
        actor=actor,
        config=_agent_config(sensor_localization=True),
    )

    specs = prepare_sensor_spawn_specs([context], _blueprint_library(), PerceptionRequirements())

    assert [spec.sensor_type for spec in specs] == [SensorType.GNSS, SensorType.IMU, SensorType.COLLISION]
    assert [spec.parent_actor_id for spec in specs] == [10, 10, 10]


def test_prepare_rsu_perception_specs_use_global_transforms_without_parent(mocker):
    _patch_perception_sensor_adapters(mocker)
    actor = SimpleNamespace(id=20)
    config = _agent_config(sensor_localization=False, camera_num=2, spawn_position=[10.0, 20.0, 5.0, 0.0, 0.0, 0.0])
    context = AgentSensorContext(
        agent_index=0,
        agent_type=AgentType.RSU,
        actor=actor,
        config=config,
    )
    requirements = PerceptionRequirements(force_rgb_camera=True, force_lidar=True, force_semantic_lidar=True)

    specs = prepare_sensor_spawn_specs([context], _blueprint_library(), requirements)

    assert [spec.sensor_type for spec in specs] == [
        SensorType.CAMERA,
        SensorType.CAMERA,
        SensorType.LIDAR,
        SensorType.SEMANTIC_LIDAR,
    ]
    assert all(spec.parent_actor_id is None for spec in specs)
    assert specs[0].transform.location.x == 10.0
    assert specs[0].transform.location.y == 20.0
    assert specs[0].transform.location.z == 6.0


def test_build_sensor_actor_bundles_preserves_camera_order(mocker):
    _patch_perception_sensor_adapters(mocker)
    contexts = [
        AgentSensorContext(
            agent_index=0,
            agent_type=AgentType.RSU,
            actor=SimpleNamespace(id=20),
            config=_agent_config(sensor_localization=True, camera_num=2),
        )
    ]
    requirements = PerceptionRequirements(force_rgb_camera=True, force_lidar=True)
    specs = prepare_sensor_spawn_specs(contexts, _blueprint_library(), requirements)
    actors = [SimpleNamespace(id=100 + index) for index in range(len(specs))]

    bundles = build_sensor_actor_bundles(1, specs, actors)

    assert bundles[0].gnss is actors[0]
    assert bundles[0].cameras == (actors[1], actors[2])
    assert bundles[0].lidar is actors[3]
    assert bundles[0].actor_ids() == [100, 101, 102, 103]
