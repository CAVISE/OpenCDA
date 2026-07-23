"""Map geometry shared by all agents in one CARLA world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import carla
import numpy as np
import numpy.typing as npt
from matplotlib.path import Path
from shapely.geometry import Polygon

from opencda.core.map.map_utils import lateral_shift, list_loc2array, list_wpt2array

LaneInfo = Mapping[str, Any]
TrafficLightInfo = Mapping[str, Any]
BoundInfo = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SharedMapData:
    """Preprocessed map geometry that is independent of an observing agent."""

    topology: tuple[carla.Waypoint, ...]
    lane_info: Mapping[str, LaneInfo]
    crosswalk_info: Mapping[str, Mapping[str, Any]]
    traffic_light_info: Mapping[str, TrafficLightInfo]
    bound_info: BoundInfo

    @classmethod
    def empty(cls) -> SharedMapData:
        """Return map data for a disabled MapManager."""
        return cls(
            topology=(),
            lane_info={},
            crosswalk_info={},
            traffic_light_info={},
            bound_info={
                "lanes": {"ids": [], "bounds": np.empty((0, 2, 2), dtype=np.float64)},
                "crosswalks": {"ids": [], "bounds": np.empty((0, 2, 2), dtype=np.float64)},
            },
        )

    @classmethod
    def build(
        cls,
        world: carla.World,
        carla_map: carla.Map,
        lane_sample_resolution: float,
    ) -> SharedMapData:
        """Preprocess topology, lanes, bounds, and traffic-light geometry."""
        if lane_sample_resolution <= 0:
            raise ValueError("lane_sample_resolution must be positive.")

        topology = tuple(
            sorted(
                (segment[0] for segment in carla_map.get_topology()),
                key=lambda waypoint: waypoint.transform.location.z,
            )
        )
        traffic_light_info = cls._build_traffic_light_info(world)
        lane_info, lane_ids, lane_bounds = cls._build_lane_info(
            topology,
            traffic_light_info,
            lane_sample_resolution,
        )

        return cls(
            topology=topology,
            lane_info=lane_info,
            crosswalk_info={},
            traffic_light_info=traffic_light_info,
            bound_info={
                "lanes": {"ids": lane_ids, "bounds": lane_bounds},
                "crosswalks": {"ids": [], "bounds": np.empty((0, 2, 2), dtype=np.float64)},
            },
        )

    @staticmethod
    def _build_traffic_light_info(world: carla.World) -> dict[str, dict[str, Any]]:
        traffic_light_info: dict[str, dict[str, Any]] = {}
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            base_transform = actor.get_transform()
            base_rotation = base_transform.rotation.yaw
            trigger_location = base_transform.transform(actor.trigger_volume.location)
            trigger_transform = carla.Transform(trigger_location, carla.Rotation(yaw=base_rotation))
            extent = actor.trigger_volume.extent
            extent_x = extent.x
            extent_y = extent.y + 0.5
            corners = np.array(
                [
                    [-extent_x, -extent_y],
                    [extent_x, -extent_y],
                    [extent_x, extent_y],
                    [-extent_x, extent_y],
                ]
            )
            for corner in corners:
                location = trigger_transform.transform(carla.Location(corner[0], corner[1]))
                corner[0] = location.x
                corner[1] = location.y

            polygon = Polygon(corners)
            traffic_light_info[str(actor.id)] = {
                "actor": actor,
                "corners": polygon,
                "path": Path(polygon.boundary.coords[:]),
                "base_rot": base_rotation,
                "base_transform": base_transform,
            }
        return traffic_light_info

    @classmethod
    def _build_lane_info(
        cls,
        topology: tuple[carla.Waypoint, ...],
        traffic_light_info: Mapping[str, TrafficLightInfo],
        lane_sample_resolution: float,
    ) -> tuple[dict[str, dict[str, Any]], list[str], npt.NDArray[np.float64]]:
        lane_info: dict[str, dict[str, Any]] = {}
        lane_ids: list[str] = []
        lane_bounds: list[npt.NDArray[np.float64]] = []

        for index, waypoint in enumerate(topology):
            lane_id = f"lane-{index}"
            waypoints = cls._sample_lane(waypoint, lane_sample_resolution)
            left_marking = [lateral_shift(item.transform, -item.lane_width * 0.5) for item in waypoints]
            right_marking = [lateral_shift(item.transform, item.lane_width * 0.5) for item in waypoints]
            left_array = list_loc2array(left_marking)
            right_array = list_loc2array(right_marking)
            mid_array = list_wpt2array(waypoints)

            lane_ids.append(lane_id)
            lane_bounds.append(cls._get_bounds(left_array, right_array))
            lane_info[lane_id] = {
                "xyz_left": left_array,
                "xyz_right": right_array,
                "xyz_mid": mid_array,
                "tl_id": cls._associate_traffic_light(mid_array, traffic_light_info),
            }

        bounds = np.concatenate(lane_bounds, axis=0) if lane_bounds else np.empty((0, 2, 2), dtype=np.float64)
        return lane_info, lane_ids, bounds

    @staticmethod
    def _sample_lane(waypoint: carla.Waypoint, resolution: float) -> list[carla.Waypoint]:
        waypoints = [waypoint]
        options = waypoint.next(resolution)
        if not options:
            return waypoints

        next_waypoint = options[0]
        while next_waypoint.road_id == waypoint.road_id and next_waypoint.lane_id == waypoint.lane_id:
            waypoints.append(next_waypoint)
            options = next_waypoint.next(resolution)
            if not options:
                break
            next_waypoint = options[0]
        return waypoints

    @staticmethod
    def _get_bounds(
        left_lane: npt.NDArray[np.float64],
        right_lane: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                [
                    [min(np.min(left_lane[:, 0]), np.min(right_lane[:, 0])), min(np.min(left_lane[:, 1]), np.min(right_lane[:, 1]))],
                    [max(np.max(left_lane[:, 0]), np.max(right_lane[:, 0])), max(np.max(left_lane[:, 1]), np.max(right_lane[:, 1]))],
                ]
            ]
        )

    @staticmethod
    def _associate_traffic_light(
        mid_lane: npt.NDArray[np.float64],
        traffic_light_info: Mapping[str, TrafficLightInfo],
    ) -> str:
        associated_id = ""
        for traffic_light_id, traffic_light in traffic_light_info.items():
            if traffic_light["path"].contains_points(mid_lane[:, :2]).any():
                associated_id = traffic_light_id
        return associated_id


@dataclass(slots=True)
class _MapDataCacheEntry:
    world: carla.World
    carla_map: carla.Map
    lane_sample_resolution: float
    data: SharedMapData


class MapDataCache:
    """Cache shared map data by CARLA world, map, and sampling resolution."""

    def __init__(self) -> None:
        self._entries: list[_MapDataCacheEntry] = []

    def get_or_build(
        self,
        world: carla.World,
        carla_map: carla.Map,
        config: Mapping[str, Any],
    ) -> SharedMapData:
        """Return matching map data or build it once for the scenario."""
        if not config["activate"]:
            return SharedMapData.empty()

        resolution = float(config["lane_sample_resolution"])
        for entry in self._entries:
            if entry.world is world and entry.carla_map is carla_map and entry.lane_sample_resolution == resolution:
                return entry.data

        data = SharedMapData.build(world, carla_map, resolution)
        self._entries.append(
            _MapDataCacheEntry(
                world=world,
                carla_map=carla_map,
                lane_sample_resolution=resolution,
                data=data,
            )
        )
        return data
