"""HDMap manager"""

from __future__ import annotations

import math
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Mapping, NoReturn, cast

import cv2
import carla
import numpy as np
import numpy.typing as npt

from opencda.core.sensing.perception.sensor_transformation import world_to_sensor
from opencda.core.map.map_data import SharedMapData
from opencda.core.map.map_utils import convert_tl_status
from opencda.core.map.map_drawing import cv2_subpixel, draw_agent, draw_road, draw_lane

if TYPE_CHECKING:
    from opencda.core.common.world_frame import WorldActorState, WorldFrame

logger = logging.getLogger("cavise.opencda.opencda.core.map.map_manager")

AgentInfo = dict[int, dict[str, Any]]


class MapManager(object):
    """
    This class is used to manage HD Map. We emulate the style of Lyft dataset.

    Parameters
    ----------
    vehicle : Carla.vehicle
        The ego vehicle.

    carla_map : Carla.Map
        The carla simulator map.

    config : dict
        All the map manager parameters.

    Attributes
    ----------
    world : carla.world
        Carla simulation world.

    center : carla.Transform
        The rasterization map's center pose.

    meter_per_pixel : float
        m/pixel

    raster_size : float
        The rasterization map size in pixels.

    raster_radius : float
        The valid radius(m) in the center of the rasterization map.

    topology : list
        Map topology in list.

    lane_info : dict
        A dictionary that contains all lane information.

    crosswalk_info : dict
        A dictionary that contains all crosswalk information.

    traffic_light_info : dict
        A dictionary that contains all traffic light information.

    bound_info : dict
        A dictionary that saves boundary information of lanes and crosswalks.
        It is used to efficiently filter out invalid lanes/crosswarlks.

    lane_sample_resolution : int
        The sampling resolution for drawing lanes.

    static_bev : np.array
        The static bev map containing lanes and drivable road information.

    dynamic_bev : np.array
        The dynamic bev map containing vehicle's information.

    vis_bev : np.array
        The comprehensive bev map for visualization.

    """

    def _abort(self, message: str) -> NoReturn:
        logger.error(message)
        raise RuntimeError(message)

    def _require_center(self) -> carla.Transform:
        if self.center is None:
            self._abort("MapManager center is not initialized. Call update_information() before rasterization.")
        return self.center

    def __init__(
        self,
        vehicle: carla.Vehicle,
        carla_map: carla.Map,
        config: Mapping[str, Any],
        shared_map_data: SharedMapData | None = None,
    ) -> None:
        self.world = vehicle.get_world()
        self.agent_id = vehicle.id
        self.carla_map = carla_map
        self.center: carla.Transform | None = None
        self._world_frame: WorldFrame | None = None

        self.activate = config["activate"]
        self.visualize = config["visualize"]
        self.pixels_per_meter = config["pixels_per_meter"]
        self.meter_per_pixel = 1 / self.pixels_per_meter
        self.raster_size = np.array([config["raster_size"][0], config["raster_size"][1]])
        self.lane_sample_resolution = config["lane_sample_resolution"]

        self.raster_radius = float(np.linalg.norm(self.raster_size * np.array([self.meter_per_pixel, self.meter_per_pixel]))) / 2

        # bev maps
        self.dynamic_bev: npt.NDArray[np.uint8] | None = None
        self.static_bev: npt.NDArray[np.uint8] | None = None
        self.vis_bev: npt.NDArray[np.uint8] | None = None

        if not self.activate:
            map_data = SharedMapData.empty()
        else:
            map_data = shared_map_data or SharedMapData.build(
                self.world,
                carla_map,
                float(self.lane_sample_resolution),
            )

        self.topology = map_data.topology
        self.lane_info = map_data.lane_info
        self.crosswalk_info = map_data.crosswalk_info
        self.traffic_light_info = map_data.traffic_light_info
        self.bound_info = map_data.bound_info

    def update_information(self, ego_pose: carla.Transform, world_frame: WorldFrame | None = None) -> None:
        """
        Update the ego pose as the map center.

        Parameters
        ----------
        ego_pose : carla.Transform
        """
        self.center = ego_pose
        self._world_frame = world_frame

    def run_step(self) -> None:
        """
        Rasterization + Visualize the bev map if needed.
        """
        if not self.activate:
            return
        self.rasterize_static()
        self.rasterize_dynamic()
        # if self.visualize:
        #     cv2.imshow('the bev map of agent %s' % self.agent_id,
        #                self.vis_bev)
        #     cv2.waitKey(1)

    def agents_in_range(self, radius: float, agents_dict: AgentInfo) -> AgentInfo:
        """
        Filter out all agents out of the radius.

        Parameters
        ----------
        radius : float
            Radius in meters

        agents_dict : dict
            Dictionary containing all dynamic agents.

        Returns
        -------
        The dictionary that only contains the agent in range.
        """
        final_agents: AgentInfo = {}

        # convert center to list format
        center_transform = self._require_center()
        center = [center_transform.location.x, center_transform.location.y]

        for agent_id, agent in agents_dict.items():
            location = agent["location"]
            distance = math.sqrt((location[0] - center[0]) ** 2 + (location[1] - center[1]) ** 2)
            if distance < radius:
                final_agents.update({agent_id: agent})

        return final_agents

    def indices_in_bounds(self, bounds: np.ndarray, half_extent: float) -> np.ndarray:
        """
        Get indices of elements for which the bounding box described by bounds
        intersects the one defined around center (square with side 2*half_side)

        Parameters
        ----------
        bounds :np.ndarray
            array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]

        half_extent : float
            half the side of the bounding box centered around center

        Returns
        -------
        np.ndarray: indices of elements inside radius from center
        """
        center_transform = self._require_center()
        x_center, y_center = center_transform.location.x, center_transform.location.y

        x_min_in = x_center > bounds[:, 0, 0] - half_extent
        y_min_in = y_center > bounds[:, 0, 1] - half_extent
        x_max_in = x_center < bounds[:, 1, 0] + half_extent
        y_max_in = y_center < bounds[:, 1, 1] + half_extent
        return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]

    def generate_lane_area(self, xyz_left: npt.NDArray[np.float64], xyz_right: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """
        Generate the lane area poly under rasterization map's center
        coordinate frame.

        Parameters
        ----------
        xyz_left : np.ndarray
            Left lanemarking of a lane, shape: (n, 3).
        xyz_right : np.ndarray
            Right lanemarking of a lane, shape: (n, 3).

        Returns
        -------
        lane_area : np.ndarray
            Combine left and right lane together to form a polygon.
        """
        lane_area = np.zeros((2, xyz_left.shape[0], 2), dtype=np.float64)
        # convert coordinates to center's coordinate frame
        xyz_left = xyz_left.T
        xyz_left = np.r_[xyz_left, [np.ones(xyz_left.shape[1])]]
        xyz_right = xyz_right.T
        xyz_right = np.r_[xyz_right, [np.ones(xyz_right.shape[1])]]

        # ego's coordinate frame
        center_transform = self._require_center()
        xyz_left = world_to_sensor(xyz_left, center_transform).T
        xyz_right = world_to_sensor(xyz_right, center_transform).T

        # to image coordinate frame
        lane_area[0] = xyz_left[:, :2]
        lane_area[1] = xyz_right[::-1, :2]
        # switch x and y
        lane_area = lane_area[..., ::-1]
        # y revert
        lane_area[:, :, 1] = -lane_area[:, :, 1]

        lane_area[:, :, 0] = lane_area[:, :, 0] * self.pixels_per_meter + self.raster_size[0] // 2
        lane_area[:, :, 1] = lane_area[:, :, 1] * self.pixels_per_meter + self.raster_size[1] // 2

        # to make more precise polygon
        lane_area_pixels = cv2_subpixel(lane_area)

        return lane_area_pixels

    def generate_agent_area(self, corners: list[list[float]]) -> npt.NDArray[np.int32]:
        """
        Convert the agent's bbx corners from world coordinates to
        rasterization coordinates.

        Parameters
        ----------
        corners : list
            The four corners of the agent's bbx under world coordinate.

        Returns
        -------
        agent four corners in image.
        """
        # (4, 3) numpy array
        corners_array = np.array(corners, dtype=np.float64)
        # for homogeneous transformation
        corners_array = corners_array.T
        corners_array = np.r_[corners_array, [np.ones(corners_array.shape[1])]]
        # convert to ego's coordinate frame
        center_transform = self._require_center()
        corners_array = world_to_sensor(corners_array, center_transform).T
        corners_array = corners_array[:, :2]

        # switch x and y
        corners_array = corners_array[..., ::-1]
        # y revert
        corners_array[:, 1] = -corners_array[:, 1]

        corners_array[:, 0] = corners_array[:, 0] * self.pixels_per_meter + self.raster_size[0] // 2
        corners_array[:, 1] = corners_array[:, 1] * self.pixels_per_meter + self.raster_size[1] // 2

        # to make more precise polygon
        corner_area = cv2_subpixel(corners_array[:, :2])

        return corner_area

    def load_agents_world(self) -> AgentInfo:
        """
        Load dynamic vehicle data from the shared frame or CARLA directly.

        Returns
        -------
        The dictionary contains all agents info in the carla world.
        """

        world_frame = self._world_frame
        if world_frame is not None:
            center = self._require_center()
            states = world_frame.nearby_vehicles(center.location, self.raster_radius)
            return {
                state.actor_id: world_frame.shared_actor_value(
                    "map-agent-info",
                    state.actor_id,
                    partial(self._world_actor_info, state),
                )
                for state in states
            }

        dynamic_agent_info: AgentInfo = {}
        for actor in self.world.get_actors().filter("vehicle.*"):
            dynamic_agent_info[actor.id] = self._actor_info(actor, actor.get_transform())
        return dynamic_agent_info

    @classmethod
    def _world_actor_info(cls, state: WorldActorState) -> dict[str, Any]:
        return cls._actor_info(state.actor, state.transform)

    @staticmethod
    def _actor_info(actor: carla.Actor, transform: carla.Transform) -> dict[str, Any]:
        extent = actor.bounding_box.extent
        corners = [
            carla.Location(x=-extent.x, y=-extent.y),
            carla.Location(x=-extent.x, y=extent.y),
            carla.Location(x=extent.x, y=extent.y),
            carla.Location(x=extent.x, y=-extent.y),
        ]
        transform.transform(corners)
        return {
            "location": [transform.location.x, transform.location.y, transform.location.z],
            "yaw": transform.rotation.yaw,
            "corners": [[corner.x, corner.y, corner.z] for corner in corners],
        }

    def rasterize_dynamic(self) -> None:
        """
        Rasterize the dynamic agents.

        Returns
        -------
        Rasterization image.
        """
        dynamic_bev = np.zeros(shape=(int(self.raster_size[1]), int(self.raster_size[0]), 3), dtype=np.uint8)
        vis_bev = self.vis_bev
        if vis_bev is None:
            self._abort("Static BEV must be rasterized before dynamic BEV.")
        # filter using half a radius from the center
        # retrieve all agents
        dynamic_agents = self.load_agents_world()
        # filter out agents out of range
        final_agents = self.agents_in_range(self.raster_radius, dynamic_agents)

        corner_list = []
        for agent_id, agent in final_agents.items():
            agent_corner = self.generate_agent_area(agent["corners"])
            corner_list.append(agent_corner)

        self.dynamic_bev = draw_agent(corner_list, dynamic_bev)
        self.vis_bev = draw_agent(corner_list, vis_bev)

    def rasterize_static(self) -> None:
        """
        Generate the static bev map.
        """
        static_bev = np.full(shape=(int(self.raster_size[1]), int(self.raster_size[0]), 3), fill_value=255, dtype=np.uint8)
        vis_bev = np.full(shape=(int(self.raster_size[1]), int(self.raster_size[0]), 3), fill_value=255, dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * np.array([self.meter_per_pixel, self.meter_per_pixel]))) / 2
        lane_indices = self.indices_in_bounds(self.bound_info["lanes"]["bounds"], raster_radius)
        lanes_area_list = []
        lane_type_list = []

        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = self.bound_info["lanes"]["ids"][lane_idx]
            lane_info = self.lane_info[lane_idx]
            xyz_left, xyz_right = lane_info["xyz_left"], lane_info["xyz_right"]

            # generate lane area
            lane_area = self.generate_lane_area(xyz_left, xyz_right)
            lanes_area_list.append(lane_area)

            # check the associated traffic light
            associated_tl_id = lane_info["tl_id"]
            if associated_tl_id:
                tl_actor = self.traffic_light_info[associated_tl_id]["actor"]
                frame_state = None if self._world_frame is None else self._world_frame.traffic_light_state(tl_actor.id)
                status = convert_tl_status(tl_actor.get_state() if frame_state is None else frame_state.state)
                lane_type_list.append(status)
            else:
                lane_type_list.append("normal")

        static_bev = draw_road(lanes_area_list, static_bev)
        static_bev = draw_lane(lanes_area_list, lane_type_list, static_bev)

        vis_bev = draw_road(lanes_area_list, vis_bev)
        vis_bev = draw_lane(lanes_area_list, lane_type_list, vis_bev)

        self.static_bev = static_bev
        self.vis_bev = cast(npt.NDArray[np.uint8], cv2.cvtColor(vis_bev, cv2.COLOR_RGB2BGR))

    def destroy(self) -> None:
        if self.visualize:
            cv2.destroyAllWindows()
