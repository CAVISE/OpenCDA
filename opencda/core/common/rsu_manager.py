"""
Basic class for RSU(Roadside Unit) management.
"""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

from opencda.core.application.behavior import create_service
from opencda.core.application.behavior.behavior_service_protocol import (
    BehaviorService,
    BehaviorServiceMessageT,
    BehaviorServiceResultT,
)
from opencda.core.common.data_dumper import DataDumper
from opencda.core.sensing.perception.perception_manager import PerceptionManager
from opencda.core.sensing.localization.rsu_localization_manager import LocalizationManager

logger = logging.getLogger("cavise.opencda.opencda.core.common.rsu_manager")


class RSUManager(object):
    """
    A class manager for RSU. Currently a RSU only has perception and
    localization modules to dump sensing information.
    TODO: add V2X module to it to enable sharing sensing information online.

    Parameters
    ----------
    carla_world : carla.World
        CARLA simulation world, we need this for blueprint creation.

    config_yaml : dict
        The configuration dictionary of the RSU.

    carla_map : carla.Map
        The CARLA simulation map.

    cav_world : opencda object
        CAV World for simulation V2X communication.

    current_time : str
        Timestamp of the simulation beginning, this is used for data dump.

    data_dumping : bool
        Indicates whether to dump sensor data during simulation.

    Attributes
    ----------
    localizer : opencda object
        The current localization manager.

    perception_manager : opencda object
        The current V2X perception manager.

    data_dumper : opencda object
        Used for dumping sensor data.
    """

    current_id = 1
    used_ids = set()

    def __init__(
        self,
        carla_world,
        config_yaml,
        carla_map,
        cav_world,
        current_time="",
        data_dumping=False,
        autogenerate_id_on_failure=True,
        behavior_services: Optional[Iterable[BehaviorService[BehaviorServiceMessageT, BehaviorServiceResultT]]] = None,
    ):
        config_id = config_yaml.get("id")

        if config_id is not None:
            try:
                id_int = int(config_id)

                if id_int < 0:
                    raise ValueError("Negative ID")
                candidate = f"rsu-{id_int}"
                if candidate in RSUManager.used_ids:
                    logger.warning(f"Duplicate RSU ID detected: {candidate!r}.")
                    raise ValueError(f"Duplicate RSU ID detected: {candidate!r}.")
                self.rid = candidate
                RSUManager.used_ids.add(self.rid)

            except (ValueError, TypeError):
                if autogenerate_id_on_failure:
                    self.rid = self.__generate_unique_rsu_id()
                    logger.warning(f"Invalid or unavailable RSU ID in config: {config_id!r}. Assigned auto-generated ID: {self.rid}")
                else:
                    logger.error(f"Invalid or unavailable RSU ID in config: {config_id!r}.")
                    raise
        else:
            if autogenerate_id_on_failure:
                self.rid = self.__generate_unique_rsu_id()
                logger.debug(f"No RSU ID specified in config. Assigned auto-generated ID: {self.rid}")
            else:
                logger.error("No RSU ID specified in config.")
                raise ValueError("No RSU ID specified in config.")

        # read map from the world everytime is time-consuming, so we need
        # explicitly extract here
        self.carla_map = carla_map

        # retrieve the configure for different modules
        # TODO: add v2x module to rsu later
        sensing_config = config_yaml["sensing"]
        sensing_config["localization"]["global_position"] = config_yaml["spawn_position"]
        sensing_config["perception"]["global_position"] = config_yaml["spawn_position"]

        # localization module
        self.localizer = LocalizationManager(carla_world, sensing_config["localization"], self.carla_map)

        # perception module
        self.perception_manager = PerceptionManager(
            vehicle=None,
            config_yaml=sensing_config["perception"],
            cav_world=cav_world,
            infra_id=self.rid,
            data_dump=data_dumping,
            carla_world=carla_world,
        )
        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager, self.rid, save_time=current_time)
        else:
            self.data_dumper = None

        if behavior_services is None:
            behavior_services = self.__build_behavior_services(config_yaml)

        self.__set_behavior_services(behavior_services)
        self.__attach_behavior_services()

        cav_world.update_rsu_manager(self)

    def __generate_unique_rsu_id(self):
        """Generates a unique RSU ID in the format 'rsu-<number>', avoiding duplicates."""
        while True:
            candidate = f"rsu-{RSUManager.current_id}"
            if candidate not in RSUManager.used_ids:
                RSUManager.used_ids.add(candidate)
                RSUManager.current_id += 1
                return candidate
            RSUManager.current_id += 1

    def __build_behavior_services(
        self, config_yaml: dict[str, Any]
    ) -> list[BehaviorService[BehaviorServiceMessageT, BehaviorServiceResultT]]:
        service_configs = config_yaml.get("behavior_services", [])
        behavior_services = []

        for service_config in service_configs:
            service_config_dict = dict(service_config)
            service_type = service_config_dict.pop("type", None)
            if service_type is None:
                raise ValueError("Each behavior service config must define 'type'.")

            behavior_services.append(create_service(service_name=service_type, **service_config_dict))
            logger.info("Attached behavior service '%s' to RSU %r.", service_type, self.rid)

        return behavior_services

    def __set_behavior_services(
        self, behavior_services: Optional[Iterable[BehaviorService[BehaviorServiceMessageT, BehaviorServiceResultT]]]
    ) -> None:
        services = tuple(behavior_services or ())
        self.__validate_behavior_services(services)
        self.behavior_services = services
        self.behavior_service_results = {}
        self._behavior_services_by_id = {service.service_id: service for service in self.behavior_services}

    def __validate_behavior_services(
        self, behavior_services: Tuple[BehaviorService[BehaviorServiceMessageT, BehaviorServiceResultT], ...]
    ) -> None:
        seen_service_ids = set()

        for service in behavior_services:
            if not isinstance(service, BehaviorService):
                raise TypeError(f"Each behavior service must implement the BehaviorService protocol; got {type(service).__name__!r}.")

            service_id = service.service_id
            if service_id in seen_service_ids:
                raise ValueError(f"Duplicate behavior service ID detected: {service_id!r}.")

            seen_service_ids.add(service_id)

    def __attach_behavior_services(self) -> None:
        attached_services = []

        try:
            for service in self.behavior_services:
                service.on_attach(self.rid)
                attached_services.append(service)
        except Exception:
            for service in reversed(attached_services):
                try:
                    service.on_detach()
                except Exception:
                    logger.exception(
                        "Failed to detach behavior service %r while rolling back RSU %r attachment.",
                        service.service_id,
                        self.rid,
                    )
            raise

    def __detach_behavior_services(self) -> None:
        first_exception = None

        for service in reversed(self.behavior_services):
            try:
                service.on_detach()
            except Exception as exc:
                logger.exception(
                    "Failed to detach behavior service %r from RSU %r.",
                    service.service_id,
                    self.rid,
                )
                if first_exception is None:
                    first_exception = exc

        if first_exception is not None:
            raise first_exception

    def __validate_behavior_service_messages(self, messages: list[BehaviorServiceMessageT]) -> None:
        for message in messages:

            service_id = getattr(message, "service_id", None)
            if not isinstance(service_id, str):
                raise TypeError(f"Each behavior service message must define a non-empty 'service_id' attribute; got {type(message).__name__!r}.")

            if service_id not in self._behavior_services_by_id:
                raise ValueError(f"Behavior service message references unknown service_id {service_id!r}.")

    def __group_behavior_service_messages(
        self, messages: list[BehaviorServiceMessageT]
    ) -> Dict[str, list[BehaviorServiceMessageT]]:
        grouped_messages = {service.service_id: [] for service in self.behavior_services}

        for message in messages:
            grouped_messages[message.service_id].append(message)

        return grouped_messages

    def update_behavior_services(self, messages: list[BehaviorServiceMessageT]) -> None:
        self.__validate_behavior_service_messages(messages)
        grouped_messages = self.__group_behavior_service_messages(messages)
        self.behavior_service_results = {}

        for service in self.behavior_services:
            service_messages = grouped_messages[service.service_id]
            self.behavior_service_results[service.service_id] = service.process(service_messages)

    def update_info(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()

        # TODO: object detection - pass it to other CAVs for V2X perception
        self.perception_manager.detect(ego_pos)

    def update_info_v2x(self):
        # TODO: Добавить обновление информации
        pass

    def run_step(self, messages: Optional[list[Any]] = None):
        """
        Run behavior services for the provided message batch and
        execute the current RSU step side effects.
        """
        if messages is not None:
            self.update_behavior_services(messages)

        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager, self.localizer, None)

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        try:
            self.__detach_behavior_services()
        finally:
            self.perception_manager.destroy()
            self.localizer.destroy()
