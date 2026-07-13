from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Mapping, NotRequired, Sequence, TypedDict, cast

import numpy as np

from opencda.core.common.misc import get_speed
from opencda.core.common.utils import transform_to_tuple
from opencda.core.sensing.perception import sensor_transformation as st

if TYPE_CHECKING:
    import carla

    from opencda.core.common.agent_manager import AgentManager
    from opencda.core.plan.behavior_agent import BehaviorAgent
    from opencda.core.sensing.localization.protocol import Localizer
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle
    from opencda.core.sensing.perception.perception_manager import PerceptionManager, SensorMeasurement

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
    spoofing_mask: NotRequired[np.ndarray]
    camera0: list[object]  # noqa: DC01


class CoperceptionDataProcessor:
    def __init__(self, sensor_sync_timeout_seconds: float = 1.0) -> None:
        self.sensor_sync_timeout_seconds = sensor_sync_timeout_seconds

    @staticmethod
    def _build_live_camera_snapshots(
        perception_manager: PerceptionManager,
        sensor_measurements: Mapping[str, "SensorMeasurement"] | None = None,
    ) -> list[object]:
        # TODO: Populate this with in-memory camera data when OpenCOOD camera-based
        # cooperative perception models are supported in the live pipeline.
        _ = perception_manager
        _ = sensor_measurements
        return []

    @staticmethod
    def _wait_for_sensor_frame(
        perception_manager: PerceptionManager,
        agent_id: str,
        frame: int,
        timeout_seconds: float,
    ) -> dict[str, "SensorMeasurement"]:
        try:
            return perception_manager.wait_for_sensor_frame(frame, timeout_seconds)
        except RuntimeError as error:
            raise RuntimeError(f"CoP sensor synchronization failed for agent '{agent_id}' on CARLA frame {frame}: {error}") from error

    @staticmethod
    def build_live_params(
        perception_manager: PerceptionManager,
        localizer: Localizer,
        actor: carla.Actor,
        behavior_agent: BehaviorAgent | None,
        sensor_measurements: Mapping[str, "SensorMeasurement"],
        *,
        is_rsu: bool,
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

        localization_state = localizer.get_state()
        predicted_ego_pos = localization_state.transform
        true_ego_pos = actor.get_transform()
        dump_yml["RSU"] = is_rsu

        dump_yml["predicted_ego_pos"] = transform_to_tuple(predicted_ego_pos)
        dump_yml["true_ego_pos"] = transform_to_tuple(true_ego_pos)
        dump_yml["ego_speed"] = float(localization_state.speed_kmh)

        if perception_manager.lidar is None:
            raise RuntimeError("Coperception requires LiDAR, but perception_manager.lidar is not initialized.")
        lidar_transform = cast("carla.Transform", sensor_measurements["lidar"].transform)
        dump_yml["lidar_pose"] = transform_to_tuple(lidar_transform)

        for i, camera in enumerate(getattr(perception_manager, "rgb_camera", None) or []):
            camera_transform = cast("carla.Transform", sensor_measurements[f"camera{i}"].transform)
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
        single_cav_list: Sequence[AgentManager],
        rsu_list: Sequence[AgentManager],
        tick_number: int,
        sensor_frame: int,
    ) -> OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None:
        timestamp = f"{tick_number:06d}"
        if len(single_cav_list) == 0 and len(rsu_list) == 0:
            logger.warning("Skipping cooperative perception tick %s because there are no CAV or RSU agents.", tick_number)
            return None

        single_batch: OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]
        single_batch = OrderedDict()

        ego_vehicle_id = single_cav_list[0].id if len(single_cav_list) > 0 else None

        for agent_manager in (*single_cav_list, *rsu_list):
            agent = agent_manager.agent
            perception_manager = agent.perception_manager
            if perception_manager.lidar is None:
                logger.warning(
                    "Skipping cooperative perception agent %s on tick %s because LiDAR is not initialized.",
                    agent_manager.id,
                    tick_number,
                )
                continue

            sensor_measurements = self._wait_for_sensor_frame(
                perception_manager,
                agent_manager.id,
                sensor_frame,
                self.sensor_sync_timeout_seconds,
            )
            lidar_measurement = sensor_measurements.get("lidar")
            lidar_data = None if lidar_measurement is None or lidar_measurement.data is None else np.asarray(lidar_measurement.data)
            if lidar_data is None:
                logger.warning(
                    "Skipping cooperative perception agent %s on tick %s because LiDAR data is not initialized.",
                    agent_manager.id,
                    tick_number,
                )
                continue

            lidar_data = cast(np.ndarray, lidar_data)
            agent_record: OrderedDict[str, LiveMemorySnapshot | bool] = OrderedDict()
            single_batch[agent_manager.id] = agent_record
            agent_record[timestamp] = {
                "params": self.build_live_params(
                    perception_manager,
                    agent.localizer,
                    agent.actor,
                    agent.behavior_agent if agent.is_vehicle else None,
                    sensor_measurements,
                    is_rsu=not agent.is_vehicle,
                ),
                "lidar_np": lidar_data.copy(),
                "camera0": self._build_live_camera_snapshots(perception_manager, sensor_measurements),
            }
            agent_record["ego"] = agent_manager.id == ego_vehicle_id

        if len(single_batch) == 0:
            logger.warning("Skipping cooperative perception tick %s because no agents have valid LiDAR data.", tick_number)
            return None

        return OrderedDict({0: single_batch})
