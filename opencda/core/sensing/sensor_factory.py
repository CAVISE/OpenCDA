"""Prepare and group CARLA sensor actors for batch creation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import carla

from opencda.core.common.agent import AgentType
from opencda.core.safety.sensors import CollisionSensor
from opencda.core.sensing.localization.factory import resolve_localization_provider
from opencda.core.sensing.localization.sensors import GnssSensor, ImuSensor
from opencda.core.sensing.perception.perception_manager import (
    CameraSensor,
    LidarSensor,
    PerceptionRequirements,
    SemanticLidarSensor,
)
from opencda.core.sensing.sensor_types import AgentSensorContext, SensorActorBundle, SensorSpawnSpec, SensorType


def prepare_sensor_spawn_specs(
    contexts: Sequence[AgentSensorContext],
    blueprint_library: Any,
    requirements: PerceptionRequirements,
) -> list[SensorSpawnSpec]:
    """Build all sensor spawn requests without contacting CARLA's spawn API."""
    spawn_specs: list[SensorSpawnSpec] = []
    for context in contexts:
        sensing_config = context.config["sensing"]
        localization_config = sensing_config["localization"]
        if resolve_localization_provider(localization_config) == "sensor":
            spawn_specs.append(
                SensorSpawnSpec(
                    agent_index=context.agent_index,
                    sensor_type=SensorType.GNSS,
                    blueprint=GnssSensor.prepare_blueprint(blueprint_library, localization_config["gnss"]),
                    transform=GnssSensor.spawn_transform(),
                    parent_actor_id=context.actor.id,
                )
            )
            if context.agent_type is AgentType.CAV:
                spawn_specs.append(
                    SensorSpawnSpec(
                        agent_index=context.agent_index,
                        sensor_type=SensorType.IMU,
                        blueprint=ImuSensor.prepare_blueprint(blueprint_library),
                        transform=ImuSensor.spawn_transform(),
                        parent_actor_id=context.actor.id,
                    )
                )

        perception_config = sensing_config["perception"]
        global_position = context.config["spawn_position"] if context.agent_type is AgentType.RSU else None
        parent_actor_id = context.actor.id if context.agent_type is AgentType.CAV else None
        activate = perception_config["activate"]
        camera_visualize = perception_config["camera"]["visualize"]
        camera_num = perception_config["camera"]["num"]
        camera_positions = perception_config["camera"]["positions"]

        if activate or camera_visualize or requirements.force_rgb_camera:
            if len(camera_positions) != camera_num:
                raise ValueError("The camera number has to be the same as the length of the relative positions list.")
            for camera_index, relative_position in enumerate(camera_positions):
                spawn_specs.append(
                    SensorSpawnSpec(
                        agent_index=context.agent_index,
                        sensor_type=SensorType.CAMERA,
                        sensor_index=camera_index,
                        blueprint=CameraSensor.prepare_blueprint(blueprint_library),
                        transform=CameraSensor.spawn_point_estimation(relative_position, global_position),
                        parent_actor_id=parent_actor_id,
                    )
                )

        lidar_config = perception_config["lidar"]
        if lidar_config["visualize"] or activate or requirements.force_lidar:
            spawn_specs.append(
                SensorSpawnSpec(
                    agent_index=context.agent_index,
                    sensor_type=SensorType.LIDAR,
                    blueprint=LidarSensor.prepare_blueprint(blueprint_library, lidar_config),
                    transform=LidarSensor.spawn_point_estimation(global_position),
                    parent_actor_id=parent_actor_id,
                )
            )

        if requirements.force_semantic_lidar:
            spawn_specs.append(
                SensorSpawnSpec(
                    agent_index=context.agent_index,
                    sensor_type=SensorType.SEMANTIC_LIDAR,
                    blueprint=SemanticLidarSensor.prepare_blueprint(blueprint_library, lidar_config),
                    transform=SemanticLidarSensor.spawn_point_estimation(global_position),
                    parent_actor_id=parent_actor_id,
                )
            )

        if context.agent_type is AgentType.CAV:
            spawn_specs.append(
                SensorSpawnSpec(
                    agent_index=context.agent_index,
                    sensor_type=SensorType.COLLISION,
                    blueprint=CollisionSensor.prepare_blueprint(blueprint_library),
                    transform=CollisionSensor.spawn_transform(),
                    parent_actor_id=context.actor.id,
                )
            )

    return spawn_specs


def build_sensor_actor_bundles(
    agent_count: int,
    spawn_specs: Sequence[SensorSpawnSpec],
    sensor_actors: Sequence[carla.Actor],
) -> list[SensorActorBundle]:
    """Group spawned sensor actors by agent and sensor role."""
    if len(spawn_specs) != len(sensor_actors):
        raise ValueError("Sensor spec and actor counts must match.")

    values: list[dict[str, Any]] = [{"cameras": []} for _ in range(agent_count)]
    for spec, actor in zip(spawn_specs, sensor_actors, strict=True):
        target = values[spec.agent_index]
        if spec.sensor_type is SensorType.CAMERA:
            target["cameras"].append((spec.sensor_index, actor))
            continue

        field_name = spec.sensor_type.value
        if target.get(field_name) is not None:
            raise ValueError(f"Duplicate {spec.sensor_type.value} sensor for agent {spec.agent_index}.")
        target[field_name] = actor

    bundles: list[SensorActorBundle] = []
    for target in values:
        cameras = tuple(actor for _, actor in sorted(target.pop("cameras")))
        bundles.append(SensorActorBundle(cameras=cameras, **target))
    return bundles
