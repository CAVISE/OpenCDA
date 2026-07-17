"""Universal OpenCDA simulation agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import carla

from opencda.core.application.behavior.types import Location

if TYPE_CHECKING:
    from opencda.core.actuation.control_manager import ControlManager
    from opencda.core.common.data_dumper import DataDumper
    from opencda.core.common.tick_profiler import TickProfiler
    from opencda.core.common.world_frame import WorldFrame
    from opencda.core.map.map_manager import MapManager
    from opencda.core.plan.behavior_agent import BehaviorAgent
    from opencda.core.safety.safety_manager import SafetyManager
    from opencda.core.sensing.localization import Localizer
    from opencda.core.sensing.perception.perception_manager import PerceptionManager

logger = logging.getLogger("cavise.opencda.opencda.core.common.agent")


class AgentType(StrEnum):
    """Supported simulation agent types."""

    CAV = "cav"
    RSU = "rsu"


@dataclass(frozen=True, slots=True)
class VehicleComponents:
    """Components used only by a movable CAV agent."""

    map_manager: MapManager
    safety_manager: SafetyManager
    behavior_agent: BehaviorAgent
    controller: ControlManager
    use_carla_autopilot: bool
    carla_autopilot_port: int


class Agent:
    """CARLA actor with the OpenCDA components required by its type."""

    def __init__(
        self,
        actor: carla.Actor,
        agent_type: AgentType,
        carla_map: carla.Map,
        localizer: Localizer,
        perception_manager: PerceptionManager,
        data_dumper: DataDumper | None,
        vehicle_components: VehicleComponents | None = None,
    ) -> None:
        if agent_type is AgentType.CAV and vehicle_components is None:
            raise ValueError("CAV agent requires vehicle components.")
        if agent_type is AgentType.RSU and vehicle_components is not None:
            raise ValueError("RSU agent cannot have vehicle components.")

        self.actor = actor
        self.agent_type = agent_type
        self.carla_map = carla_map
        self.localizer = localizer
        self.perception_manager = perception_manager
        self.data_dumper = data_dumper
        self._vehicle_components = vehicle_components
        self._destroyed = False

    @property
    def is_vehicle(self) -> bool:
        return self.agent_type is AgentType.CAV

    def _require_vehicle_components(self) -> VehicleComponents:
        components = self._vehicle_components
        if components is None:
            raise RuntimeError(f"Operation is not supported by {self.agent_type.value} agent.")
        return components

    @property
    def vehicle(self) -> carla.Vehicle:
        self._require_vehicle_components()
        return cast(carla.Vehicle, self.actor)

    @property
    def map_manager(self) -> MapManager:
        return self._require_vehicle_components().map_manager

    @property
    def safety_manager(self) -> SafetyManager:
        return self._require_vehicle_components().safety_manager

    @property
    def behavior_agent(self) -> BehaviorAgent:
        return self._require_vehicle_components().behavior_agent

    @property
    def controller(self) -> ControlManager:
        return self._require_vehicle_components().controller

    @property
    def use_carla_autopilot(self) -> bool:
        components = self._vehicle_components
        return components.use_carla_autopilot if components is not None else False

    @property
    def carla_autopilot_port(self) -> int:
        return self._require_vehicle_components().carla_autopilot_port

    def update(self, world_frame: WorldFrame | None = None, profiler: TickProfiler | None = None) -> None:
        """Refresh localization, perception, and vehicle-only components."""
        if profiler is None:
            localization_state = self.localizer.update() if world_frame is None else self.localizer.update(world_frame)
        else:
            with profiler.measure("localization"):
                localization_state = self.localizer.update() if world_frame is None else self.localizer.update(world_frame)
        ego_pos = localization_state.transform.to_carla()
        ego_speed = localization_state.speed_kmh
        if profiler is None:
            objects = self.perception_manager.detect(ego_pos) if world_frame is None else self.perception_manager.detect(ego_pos, world_frame)
        else:
            with profiler.measure("perception"):
                objects = self.perception_manager.detect(ego_pos) if world_frame is None else self.perception_manager.detect(ego_pos, world_frame)

        components = self._vehicle_components
        if components is None:
            return

        if profiler is None:
            components.map_manager.update_information(ego_pos)
            components.safety_manager.update_info(
                {
                    "ego_pos": ego_pos,
                    "ego_speed": ego_speed,
                    "objects": objects,
                    "carla_map": self.carla_map,
                    "world": self.actor.get_world(),
                    "static_bev": components.map_manager.static_bev,
                }
            )
            if not components.use_carla_autopilot:
                components.behavior_agent.update_information(ego_pos, ego_speed, objects)
            components.controller.update_info(ego_pos, ego_speed)
            return

        with profiler.measure("map"):
            components.map_manager.update_information(ego_pos)
        with profiler.measure("safety"):
            components.safety_manager.update_info(
                {
                    "ego_pos": ego_pos,
                    "ego_speed": ego_speed,
                    "objects": objects,
                    "carla_map": self.carla_map,
                    "world": self.actor.get_world(),
                    "static_bev": components.map_manager.static_bev,
                }
            )
        if not components.use_carla_autopilot:
            with profiler.measure("behavior"):
                components.behavior_agent.update_information(ego_pos, ego_speed, objects)
        with profiler.measure("control"):
            components.controller.update_info(ego_pos, ego_speed)

    def set_destination(
        self,
        start_location: Location | carla.Location,
        end_location: Location | carla.Location,
        clean: bool = False,
        end_reset: bool = True,
    ) -> None:
        """Set a vehicle route."""
        behavior_agent = self.behavior_agent
        start = self._to_carla_location(start_location)
        end = self._to_carla_location(end_location)
        behavior_agent.set_destination(start, end, clean, end_reset)

    @staticmethod
    def _to_carla_location(location: Location | carla.Location) -> carla.Location:
        if isinstance(location, carla.Location):
            return location
        return carla.Location(location.x, location.y, location.z)

    def _calculate_control(
        self,
        target_speed: float | None = None,
        target_location: Location | None = None,
    ) -> carla.VehicleControl:
        components = self._require_vehicle_components()
        components.map_manager.run_step()

        if target_location is None or target_speed is None:
            target_speed, target_location = components.behavior_agent.run_step(target_speed)

        control = components.controller.run_step(target_speed, target_location)
        if self.data_dumper is not None:
            self.data_dumper.run_step(
                self.perception_manager,
                self.localizer,
                self.vehicle,
                components.behavior_agent,
            )
        return control

    def control(
        self,
        target_speed: float | None = None,
        target_location: Location | None = None,
    ) -> None:
        """Calculate and apply one vehicle control command."""
        self.vehicle.apply_control(self._calculate_control(target_speed, target_location))

    def plan_control(self, target_speed: float | None = None) -> carla.VehicleControl:
        """Calculate control without applying it, for platoon coordination."""
        return self._calculate_control(target_speed)

    def finish_step(self) -> None:
        """Run per-tick side effects not owned by behavior services."""
        components = self._vehicle_components
        if components is not None:
            if components.use_carla_autopilot:
                components.map_manager.run_step()
            return

        if self.data_dumper is not None:
            self.data_dumper.run_step(self.perception_manager, self.localizer, self.actor, None)

    def destroy(self) -> None:
        """Destroy all agent components and its CARLA actor."""
        if self._destroyed:
            return
        self._destroyed = True

        resources = [self.perception_manager, self.localizer]
        components = self._vehicle_components
        if components is not None:
            resources.extend((components.map_manager, components.safety_manager))

        first_exception: Exception | None = None
        for resource in resources:
            try:
                resource.destroy()
            except Exception as exc:
                logger.exception("Failed to destroy agent resource %s.", type(resource).__name__)
                if first_exception is None:
                    first_exception = exc

        try:
            self.actor.destroy()
        except Exception as exc:
            logger.exception("Failed to destroy CARLA actor for %s agent.", self.agent_type.value)
            if first_exception is None:
                first_exception = exc

        if first_exception is not None:
            raise first_exception
