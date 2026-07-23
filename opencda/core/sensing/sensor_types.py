"""Shared contracts for preparing and binding CARLA sensor actors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

import carla

from opencda.core.common.agent import AgentType


class SensorType(StrEnum):
    GNSS = "gnss"
    IMU = "imu"
    CAMERA = "camera"
    LIDAR = "lidar"
    SEMANTIC_LIDAR = "semantic_lidar"
    COLLISION = "collision"


@dataclass(frozen=True, slots=True)
class AgentSensorContext:
    """Agent data needed to prepare its sensor spawn requests."""

    agent_index: int
    agent_type: AgentType
    actor: carla.Actor
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SensorSpawnSpec:
    """Prepared CARLA sensor spawn request."""

    agent_index: int
    sensor_type: SensorType
    blueprint: carla.ActorBlueprint
    transform: carla.Transform
    parent_actor_id: int | None
    sensor_index: int = 0

    @property
    def label(self) -> str:
        suffix = f"[{self.sensor_index}]" if self.sensor_type is SensorType.CAMERA else ""
        return f"agent {self.agent_index} {self.sensor_type.value}{suffix}"


@dataclass(frozen=True, slots=True)
class SensorActorBundle:
    """CARLA sensor actors owned by one OpenCDA agent."""

    gnss: carla.Actor | None = None
    imu: carla.Actor | None = None
    cameras: tuple[carla.Actor, ...] = ()
    lidar: carla.Actor | None = None
    semantic_lidar: carla.Actor | None = None
    collision: carla.Actor | None = None

    def actor_ids(self) -> list[int]:
        actors = [self.gnss, self.imu, *self.cameras, self.lidar, self.semantic_lidar, self.collision]
        return [actor.id for actor in actors if actor is not None]
