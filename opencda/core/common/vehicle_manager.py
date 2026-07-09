"""
Basic class of CAV
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Sequence, Tuple, Mapping
import carla

from opencda.core.actuation.control_manager import ControlManager
from opencda.core.application.behavior import BehaviorService, TransportMessage, BROADCAST_OWNER_ID, BROADCAST_SERVICE_TYPE, create_service
from opencda.core.application.platooning.platoon_behavior_agent import PlatooningBehaviorAgent
from opencda.core.common.v2x_manager import V2XManager
from opencda.core.sensing.localization.localization_manager import LocalizationManager
from opencda.core.sensing.perception.perception_manager import PerceptionManager, PerceptionRequirements
from opencda.core.safety.safety_manager import SafetyManager
from opencda.core.plan.behavior_agent import BehaviorAgent
from opencda.core.map.map_manager import MapManager
from opencda.core.common.data_dumper import DataDumper
from opencda.core.application.behavior.types import Location

logger = logging.getLogger("cavise.opencda.opencda.core.common.vehicle_manager")


class VehicleManager(object):
    """
    A class manager to embed different modules with vehicle together.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle. We need this class to spawn our gnss and imu sensor.

    config_yaml : dict
        The configuration dictionary of this CAV.

    application : list
        The application category, currently support:['single','platoon'].

    carla_map : carla.Map
        The CARLA simulation map.

    cav_world : opencda object
        CAV World. This is used for V2X communication simulation.

    current_time : str
        Timestamp of the simulation beginning, used for data dumping.

    data_dumping : bool
        Indicates whether to dump sensor data during simulation.

    Attributes
    ----------
    v2x_manager : opencda object
        The current V2X manager.

    localizer : opencda object
        The current localization manager.

    perception_manager : opencda object
        The current V2X perception manager.

    agent : opencda object
        The current carla agent that handles the basic behavior
         planning of ego vehicle.

    controller : opencda object
        The current control manager.

    data_dumper : opencda object
        Used for dumping sensor data.
    """

    current_cav_id = 1
    current_platoon_id = 1
    current_unknown_id = 1
    used_ids: set[str] = set()

    # TODO: application и prefix как будто бы дублируют друг друга, но не факт
    def __init__(
        self,
        vehicle: carla.Vehicle,
        config_yaml: Mapping[str, Any],
        application: Sequence[str],
        carla_map: carla.Map,
        cav_world: carla.World,
        current_time: str = "",
        perception_requirements: PerceptionRequirements | None = None,
        autogenerate_id_on_failure: bool = True,  # TODO: Link with scenario config
        prefix: str = "unknown",
    ):
        config_id = config_yaml.get("id")
        self.prefix = prefix if prefix in {"cav", "platoon"} else "unknown"

        if config_id is not None:
            try:
                id_int = int(config_id)

                if id_int < 0:
                    raise ValueError("Negative ID")
                candidate = f"{self.prefix}-{id_int}"
                if candidate in VehicleManager.used_ids:
                    logger.warning(f"Duplicate vehicle ID detected: {candidate!r}.")
                    raise ValueError(f"Duplicate vehicle ID detected: {candidate!r}.")
                self.id = candidate
                VehicleManager.used_ids.add(self.id)

            except (ValueError, TypeError):
                if autogenerate_id_on_failure:
                    self.id = self.__generate_unique_vehicle_id()
                    logger.warning(f"Invalid or unavailable vehicle ID in config: {config_id!r}. Assigned auto-generated ID: {self.id}")
                else:
                    logger.error(f"Invalid or unavailable vehicle ID in config: {config_id!r}.")
                    raise
        else:
            if autogenerate_id_on_failure:
                self.id = self.__generate_unique_vehicle_id()
                logger.debug(f"No vehicle ID specified in config. Assigned auto-generated ID: {self.id}")
            else:
                logger.error("No vehicle ID specified in config.")
                raise ValueError("No vehicle ID specified in config.")

        self.vehicle = vehicle
        self.carla_map = carla_map

        # retrieve the configure for different modules
        sensing_config = config_yaml["sensing"]
        map_config = config_yaml["map_manager"]
        behavior_config = config_yaml["behavior"]
        control_config = config_yaml["controller"]
        v2x_config = config_yaml["v2x"]
        carla_autopilot = config_yaml.get("carla_autopilot", behavior_config.get("carla_autopilot", False))
        if not isinstance(carla_autopilot, bool):
            raise ValueError("Config key 'carla_autopilot' must be a bool.")
        self.use_carla_autopilot = carla_autopilot
        self.carla_autopilot_port = int(config_yaml.get("carla_autopilot_port", behavior_config.get("carla_autopilot_port", 8000)))
        self.perception_requirements = perception_requirements or PerceptionRequirements()

        # v2x module
        self.v2x_manager = V2XManager(cav_world, v2x_config, self.id)
        # localization module
        self.localizer = LocalizationManager(vehicle, sensing_config["localization"], carla_map)
        # perception module
        self.perception_manager = PerceptionManager(
            vehicle=vehicle,
            config_yaml=sensing_config["perception"],
            cav_world=cav_world,
            infra_id=self.id,
            perception_requirements=self.perception_requirements,
        )
        # map manager
        self.map_manager = MapManager(vehicle, carla_map, map_config)
        # safety manager
        self.safety_manager = SafetyManager(vehicle=vehicle, params=config_yaml["safety_manager"])
        # behavior agent is always initialized to one of the supported implementations.
        self.agent: BehaviorAgent
        if "platoon" in application:
            platoon_config = config_yaml["platoon"]
            self.agent = PlatooningBehaviorAgent(
                vehicle,
                self,
                self.v2x_manager,
                behavior_config,
                platoon_config,
                carla_map,
            )
        else:
            self.agent = BehaviorAgent(vehicle, carla_map, behavior_config)

        # Control module
        self.controller = ControlManager(control_config)

        if self.perception_requirements.enable_data_dump:
            self.data_dumper: DataDumper | None = DataDumper(self.perception_manager, self.id, save_time=current_time)
        else:
            self.data_dumper = None

        behavior_services = self.__build_behavior_services(config_yaml)

        self.__set_behavior_services(behavior_services)
        self.__attach_behavior_services()

        if self.use_carla_autopilot:
            if self.behavior_services:
                logger.warning("Vehicle %s uses CARLA autopilot; behavior services will not be executed.", self.id)
            self.vehicle.set_autopilot(True, self.carla_autopilot_port)
            logger.info("Vehicle %s is controlled by CARLA Traffic Manager on port %s.", self.id, self.carla_autopilot_port)

        cav_world.update_vehicle_manager(self)

    def __generate_unique_vehicle_id(self) -> str:
        """Generates a unique vehicle ID based on prefix."""
        while True:
            if self.prefix == "cav":
                candidate = f"cav-{VehicleManager.current_cav_id}"
                VehicleManager.current_cav_id += 1
            elif self.prefix == "platoon":
                candidate = f"platoon-{VehicleManager.current_platoon_id}"
                VehicleManager.current_platoon_id += 1
            else:
                candidate = f"unknown-{VehicleManager.current_unknown_id}"
                VehicleManager.current_unknown_id += 1

            if candidate not in VehicleManager.used_ids:
                VehicleManager.used_ids.add(candidate)
                return candidate

    def __build_behavior_services(self, config_yaml: Mapping[str, Any]) -> list[BehaviorService[Any, Any]]:
        service_configs = config_yaml.get("behavior_services", [])
        behavior_services = []

        for service_config in service_configs:
            service_config_dict = dict(service_config)
            service_type = service_config_dict.pop("type", None)
            if service_type is None:
                raise ValueError("Each behavior service config must define 'type'.")

            behavior_services.append(create_service(service_type=service_type, **service_config_dict))
            logger.info("Attached behavior service '%s' to vehicle %r.", service_type, self.id)

        return behavior_services

    def __set_behavior_services(self, behavior_services: Optional[Iterable[BehaviorService[Any, Any]]]) -> None:
        services = tuple(behavior_services or ())
        self.__validate_behavior_services(services)
        self.behavior_services = tuple(sorted(services, key=lambda service: service.priority))
        self.behavior_service_results: list[TransportMessage] = []
        self.behavior_service_states: dict[str, Any] = {}
        self._behavior_services_by_name = {service.service_type: service for service in self.behavior_services}

    def __validate_behavior_services(self, behavior_services: Tuple[BehaviorService[Any, Any], ...]) -> None:
        seen_service_ids = set()

        for service in behavior_services:
            if not isinstance(service, BehaviorService):
                raise TypeError(f"Each behavior service must implement the BehaviorService protocol; got {type(service).__name__!r}.")

            service_type = service.service_type
            if service_type in seen_service_ids:
                raise ValueError(f"Duplicate behavior service ID detected: {service_type!r}.")

            if not isinstance(service.priority, int):
                raise TypeError(f"Behavior service {service_type!r} must define an integer priority; got {type(service.priority).__name__!r}.")

            seen_service_ids.add(service_type)

    def __attach_behavior_services(self) -> None:
        attached_services = []

        try:
            for service in self.behavior_services:
                service.on_attach(self)
                attached_services.append(service)
        except Exception:
            for service in reversed(attached_services):
                try:
                    service.on_detach()
                except Exception:
                    logger.exception(
                        "Failed to detach behavior service %r while rolling back vehicle %r attachment.",
                        service.service_type,
                        self.id,
                    )
            raise

    def __detach_behavior_services(self) -> None:
        first_exception = None

        for service in reversed(self.behavior_services):
            try:
                service.on_detach()
            except Exception as exc:
                logger.exception(
                    "Failed to detach behavior service %r from vehicle %r.",
                    service.service_type,
                    self.id,
                )
                if first_exception is None:
                    first_exception = exc

        if first_exception is not None:
            raise first_exception

    def __validate_behavior_service_messages(self, messages: list[TransportMessage]) -> list[TransportMessage]:
        valid_messages: list[TransportMessage] = []

        for message in messages:
            if not isinstance(message, TransportMessage):
                raise TypeError(f"Each behavior service message must be a TransportMessage; got {type(message).__name__!r}.")

            service_type = getattr(message, "dst_service_type", None)
            if not isinstance(service_type, str):
                raise TypeError(f"Each behavior service message must define a string 'dst_service_type'; got {type(message).__name__!r}.")

            owner_id = getattr(message, "dst_owner_id", None)
            if not isinstance(owner_id, str):
                raise TypeError(f"Each behavior service message must define a string 'dst_owner_id'; got {type(message).__name__!r}.")

            if owner_id == self.id:
                if service_type not in self._behavior_services_by_name:
                    raise ValueError(f"Behavior service message references unknown service_type {service_type!r}.")
                valid_messages.append(message)
            elif owner_id == BROADCAST_OWNER_ID and message.src_owner_id != self.id:
                if service_type in self._behavior_services_by_name or service_type == BROADCAST_SERVICE_TYPE:
                    valid_messages.append(message)

        return valid_messages

    def __group_behavior_service_messages(self, messages: list[TransportMessage]) -> Mapping[str, list[TransportMessage]]:
        grouped_messages: dict[str, list[TransportMessage]] = {service.service_type: [] for service in self.behavior_services}
        grouped_messages[BROADCAST_SERVICE_TYPE] = []

        for message in messages:
            grouped_messages[message.dst_service_type].append(message)

        return grouped_messages

    def update_behavior_services(self, messages: list[TransportMessage]) -> None:
        messages = self.__validate_behavior_service_messages(messages)
        grouped_messages = self.__group_behavior_service_messages(messages)
        self.behavior_service_results.clear()

        for service in self.behavior_services:
            service_messages = grouped_messages[service.service_type]
            if BROADCAST_SERVICE_TYPE in grouped_messages:
                service_messages += grouped_messages[BROADCAST_SERVICE_TYPE]
            result_messages = service.process(service_messages)
            self.behavior_service_states[service.service_type] = service.get_state()
            if result_messages:
                self_messages = [msg for msg in result_messages if getattr(msg, "dst_owner_id", None) == self.id]
                messages.extend(self_messages)
                grouped_messages = self.__group_behavior_service_messages(messages)
                out_messages = [msg for msg in result_messages if getattr(msg, "dst_owner_id", None) != self.id]
                self.behavior_service_results.extend(out_messages)

    def set_destination(self, start_location: Location, end_location: Location, clean: bool = False, end_reset: bool = True) -> None:
        """
        Set global route.

        Parameters
        ----------
        start_location : carla.location
            The CAV start location.

        end_location : carla.location
            The CAV destination.

        clean : bool
             Indicator of whether clean waypoint queue.

        end_reset : bool
            Indicator of whether reset the end location.

        Returns
        -------
        """
        start_location_carla = carla.Location(start_location.x, start_location.y, start_location.z)
        end_location_carla = carla.Location(end_location.x, end_location.y, end_location.z)

        self.agent.set_destination(start_location_carla, end_location_carla, clean, end_reset)

    def update_info(self) -> None:
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # object detection
        objects = self.perception_manager.detect(ego_pos)

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        # this is required by safety manager
        safety_input = {
            "ego_pos": ego_pos,
            "ego_speed": ego_spd,
            "objects": objects,
            "carla_map": self.carla_map,
            "world": self.vehicle.get_world(),
            "static_bev": self.map_manager.static_bev,
        }
        self.safety_manager.update_info(safety_input)

        # leave this for platooning for now
        self.v2x_manager.update_info(ego_pos, ego_spd)
        self.agent.update_information(ego_pos, ego_spd, objects)
        # pass position and speed info to controller
        self.controller.update_info(ego_pos, ego_spd)

    def update_info_v2x(self) -> None:  # noqa: deadcode,
        # TODO: Implement
        pass

    def control(self, target_speed: float | None = None, target_location: Location | None = None) -> None:
        # visualize the bev map if needed
        self.map_manager.run_step()
        if target_location is None or target_speed is None:
            target_speed, target_location = self.agent.run_step(target_speed)
        control = self.controller.run_step(target_speed, target_location)

        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager, self.localizer, self.agent)

        self.vehicle.apply_control(control)

    def run_step(
        self,
        messages: list[TransportMessage[Any]] = [],
    ) -> tuple[list[TransportMessage[Any]], dict[str, Any]]:
        """
        Execute one step of navigation.
        """
        if self.use_carla_autopilot:
            self.map_manager.run_step()
            return [], {}

        self.update_behavior_services(messages)

        return (self.behavior_service_results, self.behavior_service_states)

    def destroy(self) -> None:
        """
        Destroy the actor vehicle
        """
        try:
            self.__detach_behavior_services()
        finally:
            self.perception_manager.destroy()
            self.localizer.destroy()
            self.map_manager.destroy()
            self.safety_manager.destroy()
            self.vehicle.destroy()
