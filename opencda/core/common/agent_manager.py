"""Universal lifecycle and behavior-service manager for OpenCDA agents."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import carla

from opencda.core.actuation.control_manager import ControlManager
from opencda.core.application.behavior import (
    BROADCAST_OWNER_ID,
    BROADCAST_SERVICE_TYPE,
    BehaviorService,
    TransportMessage,
    create_service,
)
from opencda.core.application.platooning.platoon_behavior_agent import PlatooningBehaviorAgent
from opencda.core.common.agent import Agent, AgentType, VehicleComponents
from opencda.core.common.data_dumper import DataDumper
from opencda.core.common.v2x_manager import V2XManager
from opencda.core.map.map_manager import MapManager
from opencda.core.plan.behavior_agent import BehaviorAgent
from opencda.core.safety.safety_manager import SafetyManager
from opencda.core.sensing.localization import create_localizer
from opencda.core.sensing.perception.perception_manager import PerceptionManager, PerceptionRequirements

logger = logging.getLogger("cavise.opencda.opencda.core.common.agent_manager")


class AgentManager:
    """Create, update behavior services, and destroy one OpenCDA agent."""

    _next_ids: dict[str, int] = {"cav": 1, "platoon": 1, "rsu": 1, "unknown": 1}
    _used_ids: set[str] = set()

    def __init__(
        self,
        agent: Agent,
        agent_id: str,
        behavior_services: Iterable[BehaviorService[Any, Any]] | None = None,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string.")

        self.agent = agent
        self.id = agent_id
        self._destroyed = False
        self._set_behavior_services(behavior_services)

    @classmethod
    def create(
        cls,
        actor: carla.Actor,
        config_yaml: Mapping[str, Any],
        carla_map: carla.Map,
        cav_world: Any,
        *,
        agent_type: AgentType | str,
        application: Sequence[str] = (),
        current_time: str = "",
        perception_requirements: PerceptionRequirements | None = None,
        autogenerate_id_on_failure: bool = True,
        id_prefix: str | None = None,
    ) -> AgentManager:
        """Create either a CAV or RSU through the same construction path."""
        resolved_type = AgentType(agent_type)
        prefix = cls._resolve_id_prefix(resolved_type, id_prefix)
        agent_id = cls._allocate_id(config_yaml.get("id"), prefix, autogenerate_id_on_failure)
        requirements = perception_requirements or PerceptionRequirements()
        sensing_config = config_yaml["sensing"]

        localizer = create_localizer(
            actor,
            sensing_config["localization"],
            carla_map,
            use_imu=resolved_type is AgentType.CAV,
        )
        perception_manager = cls._create_perception_manager(
            actor,
            resolved_type,
            sensing_config["perception"],
            config_yaml,
            cav_world,
            agent_id,
            requirements,
        )
        data_dumper = DataDumper(perception_manager, agent_id, save_time=current_time) if requirements.enable_data_dump else None

        vehicle_components = None
        platooning_behavior_agent = None
        if resolved_type is AgentType.CAV:
            vehicle_components, platooning_behavior_agent = cls._create_vehicle_components(
                actor,
                config_yaml,
                application,
                carla_map,
                cav_world,
                agent_id,
            )

        agent = Agent(
            actor=actor,
            agent_type=resolved_type,
            carla_map=carla_map,
            localizer=localizer,
            perception_manager=perception_manager,
            data_dumper=data_dumper,
            vehicle_components=vehicle_components,
        )
        services = cls._build_behavior_services(config_yaml, agent_id)
        manager = cls(agent=agent, agent_id=agent_id, behavior_services=services)

        if platooning_behavior_agent is not None:
            platooning_behavior_agent.bind_agent_manager(manager)

        if agent.use_carla_autopilot:
            if manager.behavior_services:
                logger.warning(
                    "Agent %s uses CARLA autopilot; behavior services will not be executed.",
                    manager.id,
                )
            agent.vehicle.set_autopilot(True, agent.carla_autopilot_port)

        cav_world.update_agent_manager(manager)
        return manager

    @staticmethod
    def _resolve_id_prefix(agent_type: AgentType, id_prefix: str | None) -> str:
        if agent_type is AgentType.RSU:
            return "rsu"
        return id_prefix if id_prefix in {"cav", "platoon"} else "unknown"

    @classmethod
    def _allocate_id(
        cls,
        configured_id: Any,
        prefix: str,
        autogenerate_id_on_failure: bool,
    ) -> str:
        try:
            if configured_id is None:
                raise ValueError("No agent ID specified in config.")
            numeric_id = int(configured_id)
            if numeric_id < 0:
                raise ValueError("Negative agent ID.")

            candidate = f"{prefix}-{numeric_id}"
            if candidate in cls._used_ids:
                raise ValueError(f"Duplicate agent ID detected: {candidate!r}.")
            cls._used_ids.add(candidate)
            return candidate
        except (TypeError, ValueError):
            if not autogenerate_id_on_failure:
                raise

        while True:
            candidate = f"{prefix}-{cls._next_ids[prefix]}"
            cls._next_ids[prefix] += 1
            if candidate not in cls._used_ids:
                cls._used_ids.add(candidate)
                logger.warning(
                    "Invalid or unavailable agent ID %r. Assigned auto-generated ID %s.",
                    configured_id,
                    candidate,
                )
                return candidate

    @classmethod
    def reset_id_registry(cls) -> None:
        """Reset process-local IDs. Intended for isolated simulation runs and tests."""
        cls._next_ids = {"cav": 1, "platoon": 1, "rsu": 1, "unknown": 1}
        cls._used_ids = set()

    @staticmethod
    def _create_perception_manager(
        actor: carla.Actor,
        agent_type: AgentType,
        perception_config: Mapping[str, Any],
        config_yaml: Mapping[str, Any],
        cav_world: Any,
        agent_id: str,
        requirements: PerceptionRequirements,
    ) -> PerceptionManager:
        config = dict(perception_config)
        if agent_type is AgentType.RSU:
            config["global_position"] = config_yaml["spawn_position"]

        return PerceptionManager(
            vehicle=cast(carla.Vehicle, actor) if agent_type is AgentType.CAV else None,
            config_yaml=config,
            cav_world=cav_world,
            infra_id=agent_id,
            perception_requirements=requirements,
            **({"carla_world": actor.get_world()} if agent_type is AgentType.RSU else {}),
        )

    @staticmethod
    def _create_vehicle_components(
        actor: carla.Actor,
        config_yaml: Mapping[str, Any],
        application: Sequence[str],
        carla_map: carla.Map,
        cav_world: Any,
        agent_id: str,
    ) -> tuple[VehicleComponents, PlatooningBehaviorAgent | None]:
        vehicle = cast(carla.Vehicle, actor)
        behavior_config = config_yaml["behavior"]
        use_carla_autopilot = config_yaml.get("carla_autopilot", behavior_config.get("carla_autopilot", False))
        if not isinstance(use_carla_autopilot, bool):
            raise ValueError("Config key 'carla_autopilot' must be a bool.")

        autopilot_port = int(config_yaml.get("carla_autopilot_port", behavior_config.get("carla_autopilot_port", 8000)))
        v2x_manager = V2XManager(cav_world, config_yaml["v2x"], agent_id)
        map_manager = MapManager(vehicle, carla_map, config_yaml["map_manager"])
        safety_manager = SafetyManager(vehicle=vehicle, params=config_yaml["safety_manager"])

        platooning_behavior_agent = None
        if "platoon" in application:
            platooning_behavior_agent = PlatooningBehaviorAgent(
                vehicle,
                None,
                v2x_manager,
                behavior_config,
                config_yaml["platoon"],
                carla_map,
            )
            behavior_agent = platooning_behavior_agent
        else:
            behavior_agent = BehaviorAgent(vehicle, carla_map, behavior_config)

        return (
            VehicleComponents(
                v2x_manager=v2x_manager,
                map_manager=map_manager,
                safety_manager=safety_manager,
                behavior_agent=behavior_agent,
                controller=ControlManager(config_yaml["controller"]),
                use_carla_autopilot=use_carla_autopilot,
                carla_autopilot_port=autopilot_port,
            ),
            platooning_behavior_agent,
        )

    @staticmethod
    def _build_behavior_services(
        config_yaml: Mapping[str, Any],
        agent_id: str,
    ) -> list[BehaviorService[Any, Any]]:
        services: list[BehaviorService[Any, Any]] = []
        for service_config in config_yaml.get("behavior_services", []):
            service_args = dict(service_config)
            service_type = service_args.pop("type", None)
            if service_type is None:
                raise ValueError("Each behavior service config must define 'type'.")
            services.append(create_service(service_type=service_type, **service_args))
            logger.info("Attached behavior service %r to agent %r.", service_type, agent_id)
        return services

    def _set_behavior_services(
        self,
        behavior_services: Iterable[BehaviorService[Any, Any]] | None,
    ) -> None:
        services = tuple(behavior_services or ())
        self._validate_behavior_services(services)
        self.behavior_services = tuple(sorted(services, key=lambda service: service.priority))
        self.behavior_service_results: list[TransportMessage[Any]] = []
        self.behavior_service_states: dict[str, Any] = {}
        self._behavior_services_by_name = {service.service_type: service for service in self.behavior_services}
        self._attach_behavior_services()

    @staticmethod
    def _validate_behavior_services(behavior_services: tuple[BehaviorService[Any, Any], ...]) -> None:
        seen_service_types: set[str] = set()
        for service in behavior_services:
            if not isinstance(service, BehaviorService):
                raise TypeError(f"Each behavior service must implement the BehaviorService protocol; got {type(service).__name__!r}.")
            if service.service_type in seen_service_types:
                raise ValueError(f"Duplicate behavior service ID detected: {service.service_type!r}.")
            if not isinstance(service.priority, int):
                raise TypeError(
                    f"Behavior service {service.service_type!r} must define an integer priority; got {type(service.priority).__name__!r}."
                )
            seen_service_types.add(service.service_type)

    def _attach_behavior_services(self) -> None:
        attached_services: list[BehaviorService[Any, Any]] = []
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
                        "Failed to detach behavior service %r while rolling back agent %r attachment.",
                        service.service_type,
                        self.id,
                    )
            raise

    def _detach_behavior_services(self) -> Exception | None:
        first_exception: Exception | None = None
        for service in reversed(self.behavior_services):
            try:
                service.on_detach()
            except Exception as exc:
                logger.exception("Failed to detach behavior service %r from agent %r.", service.service_type, self.id)
                if first_exception is None:
                    first_exception = exc
        return first_exception

    def _validate_messages(
        self,
        messages: Sequence[TransportMessage[Any]],
    ) -> list[TransportMessage[Any]]:
        valid_messages: list[TransportMessage[Any]] = []
        for message in messages:
            if not isinstance(message, TransportMessage):
                raise TypeError(f"Each behavior service message must be a TransportMessage; got {type(message).__name__!r}.")
            if not isinstance(message.dst_service_type, str):
                raise TypeError("Each behavior service message must define a string 'dst_service_type'.")
            if not isinstance(message.dst_owner_id, str):
                raise TypeError("Each behavior service message must define a string 'dst_owner_id'.")

            if message.dst_owner_id == self.id:
                if message.dst_service_type not in self._behavior_services_by_name:
                    raise ValueError(f"Behavior service message references unknown service_type {message.dst_service_type!r}.")
                valid_messages.append(message)
            elif message.dst_owner_id == BROADCAST_OWNER_ID and message.src_owner_id != self.id:
                if message.dst_service_type in self._behavior_services_by_name or message.dst_service_type == BROADCAST_SERVICE_TYPE:
                    valid_messages.append(message)
        return valid_messages

    def _group_messages(
        self,
        messages: Sequence[TransportMessage[Any]],
    ) -> dict[str, list[TransportMessage[Any]]]:
        grouped = {service.service_type: [] for service in self.behavior_services}
        grouped[BROADCAST_SERVICE_TYPE] = []
        for message in messages:
            grouped[message.dst_service_type].append(message)
        return grouped

    def update_behavior_services(
        self,
        messages: Sequence[TransportMessage[Any]],
    ) -> tuple[list[TransportMessage[Any]], dict[str, Any]]:
        """Process one message batch in behavior-service priority order."""
        if self.agent.use_carla_autopilot:
            self.behavior_service_results.clear()
            self.behavior_service_states.clear()
            return self.behavior_service_results, self.behavior_service_states

        valid_messages = self._validate_messages(messages)
        grouped_messages = self._group_messages(valid_messages)
        self.behavior_service_results.clear()

        for service in self.behavior_services:
            service_messages = grouped_messages[service.service_type] + grouped_messages[BROADCAST_SERVICE_TYPE]
            result_messages = service.process(service_messages)
            self.behavior_service_states[service.service_type] = service.get_state()
            if not result_messages:
                continue

            valid_messages.extend(message for message in result_messages if message.dst_owner_id == self.id)
            grouped_messages = self._group_messages(valid_messages)
            self.behavior_service_results.extend(message for message in result_messages if message.dst_owner_id != self.id)

        return self.behavior_service_results, self.behavior_service_states

    def destroy(self) -> None:
        """Detach behavior services and destroy the managed agent."""
        if self._destroyed:
            return
        self._destroyed = True

        first_exception = self._detach_behavior_services()
        try:
            self.agent.destroy()
        except Exception as exc:
            if first_exception is None:
                first_exception = exc

        if first_exception is not None:
            raise first_exception
