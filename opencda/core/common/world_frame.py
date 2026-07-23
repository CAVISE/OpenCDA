"""Shared CARLA world state captured once per simulation tick."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar, cast

import carla

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class WorldActorState:
    """Actor state read from a CARLA world snapshot."""

    actor_id: int
    type_id: str
    actor: carla.Actor
    transform: carla.Transform
    velocity: carla.Vector3D

    @property
    def location(self) -> carla.Location:
        return self.transform.location

    @property
    def is_vehicle(self) -> bool:
        return self.type_id.startswith("vehicle.")

    @property
    def is_walker(self) -> bool:
        return self.type_id.startswith("walker.")


@dataclass(frozen=True, slots=True)
class WorldTrafficLightState:
    """Cached static geometry and current state of a CARLA traffic light."""

    actor_id: int
    actor: carla.Actor
    state: carla.TrafficLightState
    road_id: int
    forward_vector: carla.Vector3D
    intersection_location: carla.Location


@dataclass(frozen=True, slots=True)
class _TrafficLightGeometry:
    actor_id: int
    actor: carla.Actor
    road_id: int
    forward_vector: carla.Vector3D
    intersection_location: carla.Location


class _TrafficLightGeometryCache:
    """Cache map-dependent traffic-light geometry for one CARLA world."""

    def __init__(self, carla_map: carla.Map) -> None:
        self._map = carla_map
        self._geometry: dict[int, _TrafficLightGeometry] = {}

    def capture(self, actor: carla.Actor) -> WorldTrafficLightState:
        geometry = self._geometry.get(actor.id)
        if geometry is None:
            geometry = self._build_geometry(actor)
            self._geometry[actor.id] = geometry

        return WorldTrafficLightState(
            actor_id=geometry.actor_id,
            actor=geometry.actor,
            state=actor.get_state(),
            road_id=geometry.road_id,
            forward_vector=geometry.forward_vector,
            intersection_location=geometry.intersection_location,
        )

    def _build_geometry(self, actor: carla.Actor) -> _TrafficLightGeometry:
        from opencda.core.sensing.perception.static_obstacle import TrafficLight

        trigger_location = TrafficLight.get_trafficlight_trigger_location(actor)
        trigger_waypoint = self._map.get_waypoint(trigger_location)
        road_id = trigger_waypoint.road_id
        forward_vector = trigger_waypoint.transform.get_forward_vector()
        intersection_waypoint = trigger_waypoint

        while not intersection_waypoint.is_intersection:
            next_waypoints = intersection_waypoint.next(0.5)
            if not next_waypoints:
                break
            next_waypoint = next_waypoints[0]
            if not next_waypoint.is_intersection:
                intersection_waypoint = next_waypoint
            else:
                break

        return _TrafficLightGeometry(
            actor_id=actor.id,
            actor=actor,
            road_id=road_id,
            forward_vector=forward_vector,
            intersection_location=intersection_waypoint.transform.location,
        )


class WorldFrameBuilder:
    """Capture world frames while retaining world-static derived data."""

    def __init__(self, world: carla.World, carla_map: carla.Map, cell_size: float = 50.0) -> None:
        self._world = world
        self._cell_size = cell_size
        self._traffic_light_geometry = _TrafficLightGeometryCache(carla_map)

    def capture(self, frame: int | None = None) -> "WorldFrame":
        return WorldFrame.capture(
            self._world,
            frame=frame,
            cell_size=self._cell_size,
            traffic_light_geometry=self._traffic_light_geometry,
        )


class WorldFrame:
    """A per-tick actor-state cache with a uniform-grid spatial index."""

    def __init__(
        self,
        frame: int,
        timestamp: float | None,
        actor_states: dict[int, WorldActorState],
        traffic_lights: tuple[carla.Actor, ...],
        traffic_light_states: tuple[WorldTrafficLightState, ...],
        cell_size: float,
    ) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive.")

        self.frame = frame
        self.timestamp = timestamp
        self.traffic_lights = traffic_lights
        self.traffic_light_states = traffic_light_states
        self._actor_states = actor_states
        self._cell_size = cell_size
        self._dynamic_grid: dict[tuple[int, int], tuple[WorldActorState, ...]] = self._build_dynamic_grid(actor_states.values())
        self._shared_actor_values: dict[tuple[str, int], object] = {}

    @classmethod
    def capture(
        cls,
        world: carla.World,
        frame: int | None = None,
        cell_size: float = 50.0,
        traffic_light_geometry: _TrafficLightGeometryCache | None = None,
    ) -> "WorldFrame":
        """Capture actor states without per-actor RPC calls."""
        snapshot = world.get_snapshot()
        snapshot_frame = int(snapshot.frame)
        if frame is not None and frame != snapshot_frame:
            raise RuntimeError(f"CARLA snapshot frame {snapshot_frame} does not match tick frame {frame}.")

        timestamp_data = getattr(snapshot, "timestamp", None)
        elapsed_seconds = getattr(timestamp_data, "elapsed_seconds", None)
        timestamp = float(elapsed_seconds) if elapsed_seconds is not None else None

        actor_states: dict[int, WorldActorState] = {}
        traffic_lights: list[carla.Actor] = []
        traffic_light_states: list[WorldTrafficLightState] = []

        for actor in world.get_actors():
            type_id = actor.type_id
            if type_id.startswith("traffic.traffic_light"):
                traffic_lights.append(actor)
                if traffic_light_geometry is not None:
                    traffic_light_states.append(traffic_light_geometry.capture(actor))
            if type_id.startswith("sensor."):
                continue

            actor_snapshot = snapshot.find(actor.id)
            if actor_snapshot is None:
                continue

            actor_states[actor.id] = WorldActorState(
                actor_id=actor.id,
                type_id=type_id,
                actor=actor,
                transform=actor_snapshot.get_transform(),
                velocity=actor_snapshot.get_velocity(),
            )

        return cls(
            frame=snapshot_frame,
            timestamp=timestamp,
            actor_states=actor_states,
            traffic_lights=tuple(traffic_lights),
            traffic_light_states=tuple(traffic_light_states),
            cell_size=cell_size,
        )

    def actor_state(self, actor_id: int) -> WorldActorState:
        """Return a cached actor state or fail on an inconsistent frame."""
        try:
            return self._actor_states[actor_id]
        except KeyError as exc:
            raise KeyError(f"Actor {actor_id} is absent from CARLA frame {self.frame}.") from exc

    def shared_actor_value(self, namespace: str, actor_id: int, factory: Callable[[], T]) -> T:
        """Return a frame-local actor value that is independent of the observing agent."""
        key = (namespace, actor_id)
        if key not in self._shared_actor_values:
            self._shared_actor_values[key] = factory()
        return cast(T, self._shared_actor_values[key])

    def nearby_dynamic(
        self,
        location: carla.Location,
        radius: float,
        exclude_actor_id: int | None = None,
    ) -> tuple[WorldActorState, ...]:
        """Return nearby vehicles and walkers ordered by actor id."""
        return self._nearby(location, radius, exclude_actor_id, vehicles_only=False)

    def nearby_vehicles(
        self,
        location: carla.Location,
        radius: float,
        exclude_actor_id: int | None = None,
    ) -> tuple[WorldActorState, ...]:
        """Return nearby vehicles ordered by actor id."""
        return self._nearby(location, radius, exclude_actor_id, vehicles_only=True)

    def _build_dynamic_grid(
        self,
        states: Iterable[WorldActorState],
    ) -> dict[tuple[int, int], tuple[WorldActorState, ...]]:
        mutable_grid: dict[tuple[int, int], list[WorldActorState]] = {}
        for state in states:
            if not (state.is_vehicle or state.is_walker):
                continue
            mutable_grid.setdefault(self._cell(state.location), []).append(state)
        return {cell: tuple(cell_states) for cell, cell_states in mutable_grid.items()}

    def _nearby(
        self,
        location: carla.Location,
        radius: float,
        exclude_actor_id: int | None,
        vehicles_only: bool,
    ) -> tuple[WorldActorState, ...]:
        if radius < 0:
            raise ValueError("radius must be non-negative.")

        min_x = math.floor((location.x - radius) / self._cell_size)
        max_x = math.floor((location.x + radius) / self._cell_size)
        min_y = math.floor((location.y - radius) / self._cell_size)
        max_y = math.floor((location.y + radius) / self._cell_size)
        radius_squared = radius * radius
        matches: list[WorldActorState] = []

        for cell_x in range(min_x, max_x + 1):
            for cell_y in range(min_y, max_y + 1):
                for state in self._dynamic_grid.get((cell_x, cell_y), ()):
                    if state.actor_id == exclude_actor_id or (vehicles_only and not state.is_vehicle):
                        continue
                    dx = state.location.x - location.x
                    dy = state.location.y - location.y
                    dz = state.location.z - location.z
                    if dx * dx + dy * dy + dz * dz < radius_squared:
                        matches.append(state)

        matches.sort(key=lambda state: state.actor_id)
        return tuple(matches)

    def _cell(self, location: carla.Location) -> tuple[int, int]:
        return math.floor(location.x / self._cell_size), math.floor(location.y / self._cell_size)
