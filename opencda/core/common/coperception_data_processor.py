from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence, TypeAlias, cast

import numpy as np

from opencda.core.common.misc import get_speed
from opencda.core.sensing.perception import sensor_transformation as st

if TYPE_CHECKING:
    import carla
    from opencda.core.common.rsu_manager import RSUManager
    from opencda.core.common.vehicle_manager import VehicleManager
    from opencda.core.plan.behavior_agent import BehaviorAgent
    from opencda.core.sensing.localization.localization_manager import LocalizationManager as VehicleLocalizationManager
    from opencda.core.sensing.localization.rsu_localization_manager import LocalizationManager as RsuLocalizationManager
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle
    from opencda.core.sensing.perception.perception_manager import PerceptionManager

logger = logging.getLogger("cavise.opencda.opencda.core.common.coperception_data_processor")

if TYPE_CHECKING:
    LocalizationManagerLike: TypeAlias = VehicleLocalizationManager | RsuLocalizationManager


class CoperceptionDataProcessor:
    @staticmethod
    def _build_live_camera_snapshots(perception_manager: PerceptionManager) -> list[object]:
        # TODO: Populate this with in-memory camera data when OpenCOOD camera-based
        # cooperative perception models are supported in the live pipeline.
        _ = perception_manager
        return []

    @staticmethod
    def _transform_to_list(transform: carla.Transform) -> list[float]:
        return [
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.roll,
            transform.rotation.yaw,
            transform.rotation.pitch,
        ]

    def build_live_params(
        self,
        perception_manager: PerceptionManager,
        localization_manager: LocalizationManagerLike,
        behavior_agent: BehaviorAgent | None,
    ) -> dict[str, object]:
        dump_yml: dict[str, object] = {}
        vehicle_dict: dict[int, dict[str, object]] = {}

        objects = getattr(perception_manager, "objects", {}) or {}
        vehicle_list = cast("Sequence[ObstacleVehicle]", objects.get("vehicles", []))
        for veh in vehicle_list:
            veh_carla_id = veh.carla_id
            if veh_carla_id == -1:
                continue

            veh_pos = veh.get_transform()
            veh_bbx = veh.bounding_box
            veh_speed = get_speed(veh)

            vehicle_dict[veh_carla_id] = {
                "bp_id": veh.type_id,
                "color": veh.color,
                "location": [veh_pos.location.x, veh_pos.location.y, veh_pos.location.z],
                "center": [veh_bbx.location.x, veh_bbx.location.y, veh_bbx.location.z],
                "angle": [veh_pos.rotation.roll, veh_pos.rotation.yaw, veh_pos.rotation.pitch],
                "extent": [veh_bbx.extent.x, veh_bbx.extent.y, veh_bbx.extent.z],
                "speed": veh_speed,
            }

        dump_yml["vehicles"] = vehicle_dict

        predicted_ego_pos = localization_manager.get_ego_pos()
        if hasattr(localization_manager, "vehicle"):
            vehicle_localizer = cast("VehicleLocalizationManager", localization_manager)
            true_ego_pos = vehicle_localizer.vehicle.get_transform()
        else:
            rsu_localizer = cast("RsuLocalizationManager", localization_manager)
            true_ego_pos = rsu_localizer.true_ego_pos
        dump_yml["predicted_ego_pos"] = self._transform_to_list(predicted_ego_pos)
        dump_yml["true_ego_pos"] = self._transform_to_list(true_ego_pos)
        dump_yml["ego_speed"] = float(localization_manager.get_ego_spd())

        lidar = perception_manager.lidar
        if lidar is None:
            raise RuntimeError("Coperception requires LiDAR, but perception_manager.lidar is not initialized.")
        lidar_transform = lidar.sensor.get_transform()
        dump_yml["lidar_pose"] = self._transform_to_list(lidar_transform)

        for i, camera in enumerate(getattr(perception_manager, "rgb_camera", None) or []):
            camera_transform = camera.sensor.get_transform()
            camera_intrinsic = st.get_camera_intrinsic(camera.sensor)
            lidar2world = st.x_to_world_transformation(lidar_transform)
            camera2world = st.x_to_world_transformation(camera_transform)
            world2camera = np.linalg.inv(camera2world)
            lidar2camera = np.dot(world2camera, lidar2world)
            dump_yml[f"camera{i}"] = {
                "cords": self._transform_to_list(camera_transform),
                "intrinsic": camera_intrinsic.tolist(),
                "extrinsic": lidar2camera.tolist(),
            }

        dump_yml["RSU"] = True
        if behavior_agent is not None:
            trajectory_deque = behavior_agent.get_local_planner().get_trajectory()
            dump_yml["plan_trajectory"] = [[wp.location.x, wp.location.y, spd] for wp, spd in list(trajectory_deque)]
            dump_yml["RSU"] = False

        return dump_yml

    def build_live_memory(
        self,
        single_cav_list: Sequence[VehicleManager],
        rsu_list: Sequence[RSUManager],
        tick_number: int,
    ) -> OrderedDict[int, OrderedDict[str, object]] | None:
        timestamp = f"{tick_number:06d}"
        if len(single_cav_list) == 0 and len(rsu_list) == 0:
            logger.warning("Skipping cooperative perception tick %s because there are no CAV or RSU agents.", tick_number)
            return None

        scenario_data: OrderedDict[int, OrderedDict[str, object]] = OrderedDict()
        scenario_data[0] = OrderedDict()

        ego_vehicle_id = single_cav_list[0].id if len(single_cav_list) > 0 else None

        for vehicle_manager in single_cav_list:
            vehicle_lidar = vehicle_manager.perception_manager.lidar
            if vehicle_lidar is None:
                logger.warning(
                    "Skipping cooperative perception agent %s on tick %s because LiDAR is not initialized.",
                    vehicle_manager.id,
                    tick_number,
                )
                continue
            if vehicle_lidar.data is None:
                logger.warning(
                    "Skipping cooperative perception agent %s on tick %s because LiDAR data is not initialized.",
                    vehicle_manager.id,
                    tick_number,
                )
                continue
            vehicle_lidar_data = cast(np.ndarray, vehicle_lidar.data)
            agent_record: OrderedDict[str, object] = OrderedDict()
            scenario_data[0][vehicle_manager.id] = agent_record
            agent_record[timestamp] = {
                "params": self.build_live_params(
                    vehicle_manager.perception_manager,
                    vehicle_manager.localizer,
                    vehicle_manager.agent,
                ),
                "lidar_np": vehicle_lidar_data.copy(),
                "camera0": self._build_live_camera_snapshots(vehicle_manager.perception_manager),
            }
            agent_record["ego"] = vehicle_manager.id == ego_vehicle_id

        for rsu_manager in rsu_list:
            rsu_lidar = rsu_manager.perception_manager.lidar
            if rsu_lidar is None:
                logger.warning(
                    "Skipping cooperative perception agent %s on tick %s because LiDAR is not initialized.",
                    rsu_manager.id,
                    tick_number,
                )
                continue
            if rsu_lidar.data is None:
                logger.warning(
                    "Skipping cooperative perception agent %s on tick %s because LiDAR data is not initialized.",
                    rsu_manager.id,
                    tick_number,
                )
                continue
            rsu_lidar_data = cast(np.ndarray, rsu_lidar.data)
            rsu_record: OrderedDict[str, object] = OrderedDict()
            scenario_data[0][rsu_manager.id] = rsu_record
            rsu_record[timestamp] = {
                "params": self.build_live_params(
                    rsu_manager.perception_manager,
                    rsu_manager.localizer,
                    None,
                ),
                "lidar_np": rsu_lidar_data.copy(),
                "camera0": self._build_live_camera_snapshots(rsu_manager.perception_manager),
            }
            rsu_record["ego"] = False

        if len(scenario_data[0]) == 0:
            logger.warning("Skipping cooperative perception tick %s because no agents have valid LiDAR data.", tick_number)
            return None

        return scenario_data
