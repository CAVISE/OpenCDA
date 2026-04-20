from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence, TypedDict, cast

import numpy as np

from opencda.core.common.misc import get_speed
from opencda.core.common.utils import transform_to_tuple
from opencda.core.sensing.perception import sensor_transformation as st

if TYPE_CHECKING:
    from opencda.core.common.rsu_manager import RSUManager
    from opencda.core.common.vehicle_manager import VehicleManager
    from opencda.core.plan.behavior_agent import BehaviorAgent
    from opencda.core.sensing.localization.localization_manager import LocalizationManager as VehicleLocalizationManager
    from opencda.core.sensing.localization.rsu_localization_manager import LocalizationManager as RsuLocalizationManager
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle
    from opencda.core.sensing.perception.perception_manager import PerceptionManager

logger = logging.getLogger("cavise.opencda.opencda.core.common.coperception_data_processor")


class VehicleDumpRecord(TypedDict):
    bp_id: str  # noqa: DC01
    color: str | None
    location: tuple[float, float, float]
    center: tuple[float, float, float]
    angle: tuple[float, float, float]
    extent: tuple[float, float, float]
    speed: float


class CameraDumpRecord(TypedDict):
    cords: tuple[float, float, float, float, float, float]
    intrinsic: list[list[float]]  # noqa: DC01
    extrinsic: list[list[float]]  # noqa: DC01


class LiveParams(TypedDict, total=False):
    vehicles: dict[int, "VehicleDumpRecord"]
    predicted_ego_pos: tuple[float, float, float, float, float, float]
    true_ego_pos: tuple[float, float, float, float, float, float]
    ego_speed: float
    lidar_pose: tuple[float, float, float, float, float, float]
    RSU: bool  # noqa: DC01
    plan_trajectory: list[tuple[float, float, float]]  # noqa: DC01


class LiveMemorySnapshot(TypedDict):
    params: LiveParams
    lidar_np: np.ndarray
    camera0: list[object]  # noqa: DC01


class CoperceptionDataProcessor:
    @staticmethod
    def _build_live_camera_snapshots(perception_manager: PerceptionManager) -> list[object]:
        # TODO: Populate this with in-memory camera data when OpenCOOD camera-based
        # cooperative perception models are supported in the live pipeline.
        _ = perception_manager
        return []

    @staticmethod
    def build_live_params(
        perception_manager: PerceptionManager,
        localization_manager: VehicleLocalizationManager | RsuLocalizationManager,
        behavior_agent: BehaviorAgent | None,
    ) -> LiveParams:
        dump_yml: LiveParams = {}
        vehicle_dict: dict[int, "VehicleDumpRecord"] = {}
        camera_records: dict[str, "CameraDumpRecord"] = {}

        objects = perception_manager.objects
        vehicle_list = cast("Sequence[ObstacleVehicle]", objects.get("vehicles", []))

        # NOTE: carla_id == -1 marks a perception-only detection that is not linked to a real CARLA actor.
        for veh in vehicle_list:
            veh_carla_id = veh.carla_id
            if veh_carla_id == -1:
                continue

            veh_pos = veh.get_transform()
            veh_bbx = veh.bounding_box
            veh_speed = get_speed(veh)

            vehicle_record: VehicleDumpRecord = {
                "bp_id": veh.type_id,
                "color": veh.color,
                "location": (veh_pos.location.x, veh_pos.location.y, veh_pos.location.z),
                "center": (veh_bbx.location.x, veh_bbx.location.y, veh_bbx.location.z),
                "angle": (veh_pos.rotation.roll, veh_pos.rotation.yaw, veh_pos.rotation.pitch),
                "extent": (veh_bbx.extent.x, veh_bbx.extent.y, veh_bbx.extent.z),
                "speed": veh_speed,
            }
            vehicle_dict[veh_carla_id] = vehicle_record

        dump_yml["vehicles"] = vehicle_dict

        predicted_ego_pos = localization_manager.get_ego_pos()
        if hasattr(localization_manager, "rsu"):
            rsu_localizer = cast("RsuLocalizationManager", localization_manager)
            true_ego_pos = rsu_localizer.true_ego_pos
            dump_yml["RSU"] = True
        elif hasattr(localization_manager, "vehicle"):
            vehicle_localizer = cast("VehicleLocalizationManager", localization_manager)
            true_ego_pos = vehicle_localizer.vehicle.get_transform()
            dump_yml["RSU"] = False
        else:
            raise ValueError("Unknown localization manager type")

        dump_yml["predicted_ego_pos"] = transform_to_tuple(predicted_ego_pos)
        dump_yml["true_ego_pos"] = transform_to_tuple(true_ego_pos)
        dump_yml["ego_speed"] = float(localization_manager.get_ego_spd())

        if (lidar := perception_manager.lidar) is None:
            raise RuntimeError("Coperception requires LiDAR, but perception_manager.lidar is not initialized.")
        lidar_transform = lidar.sensor.get_transform()
        dump_yml["lidar_pose"] = transform_to_tuple(lidar_transform)

        for i, camera in enumerate(getattr(perception_manager, "rgb_camera", None) or []):
            camera_transform = camera.sensor.get_transform()
            camera_intrinsic = st.get_camera_intrinsic(camera.sensor)
            lidar2world = st.x_to_world_transformation(lidar_transform)
            camera2world = st.x_to_world_transformation(camera_transform)
            world2camera = np.linalg.inv(camera2world)
            lidar2camera = np.dot(world2camera, lidar2world)
            camera_record: CameraDumpRecord = {
                "cords": transform_to_tuple(camera_transform),
                "intrinsic": cast(list[list[float]], camera_intrinsic.tolist()),
                "extrinsic": cast(list[list[float]], lidar2camera.tolist()),
            }
            camera_records[f"camera{i}"] = camera_record

        if behavior_agent is not None:
            trajectory_deque = behavior_agent.get_local_planner().get_trajectory()
            dump_yml["plan_trajectory"] = [(wp.location.x, wp.location.y, spd) for wp, spd in list(trajectory_deque)]

        return cast(LiveParams, {**dump_yml, **camera_records})

    def build_live_memory(
        self,
        single_cav_list: Sequence[VehicleManager],
        rsu_list: Sequence[RSUManager],
        tick_number: int,
    ) -> OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None:
        timestamp = f"{tick_number:06d}"
        if len(single_cav_list) == 0 and len(rsu_list) == 0:
            logger.warning("Skipping cooperative perception tick %s because there are no CAV or RSU agents.", tick_number)
            return None

        single_batch: OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]
        single_batch = OrderedDict()

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
            agent_record: OrderedDict[str, LiveMemorySnapshot | bool] = OrderedDict()
            single_batch[vehicle_manager.id] = agent_record
            agent_snapshot: LiveMemorySnapshot = {
                "params": self.build_live_params(
                    vehicle_manager.perception_manager,
                    vehicle_manager.localizer,
                    vehicle_manager.agent,
                ),
                "lidar_np": vehicle_lidar_data.copy(),
                "camera0": self._build_live_camera_snapshots(vehicle_manager.perception_manager),
            }
            agent_record[timestamp] = agent_snapshot
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
            rsu_record: OrderedDict[str, LiveMemorySnapshot | bool] = OrderedDict()
            single_batch[rsu_manager.id] = rsu_record
            rsu_snapshot: LiveMemorySnapshot = {
                "params": self.build_live_params(
                    rsu_manager.perception_manager,
                    rsu_manager.localizer,
                    None,
                ),
                "lidar_np": rsu_lidar_data.copy(),
                "camera0": self._build_live_camera_snapshots(rsu_manager.perception_manager),
            }
            rsu_record[timestamp] = rsu_snapshot
            rsu_record["ego"] = False

        if len(single_batch) == 0:
            logger.warning("Skipping cooperative perception tick %s because no agents have valid LiDAR data.", tick_number)
            return None

        return OrderedDict({0: single_batch})
