"""
Basic class for RSU(Roadside Unit) management.
"""

import logging
from dataclasses import is_dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from opencda.core.application.behavior.behavior_application_protocol import BehaviorApplication
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
        behavior_applications: Optional[Iterable[BehaviorApplication[Any, Any]]] = None,
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

        self.__set_behavior_applications(behavior_applications)
        self.__attach_behavior_applications()

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

    def __set_behavior_applications(self, behavior_applications: Optional[Iterable[BehaviorApplication[Any, Any]]]) -> None:
        applications = tuple(behavior_applications or ())
        self.__validate_behavior_applications(applications)
        self.behavior_applications = applications
        self.behavior_application_results = {}
        self._behavior_applications_by_id = {
            application.application_id: application for application in self.behavior_applications
        }

    def __validate_behavior_applications(
        self, behavior_applications: Tuple[BehaviorApplication[Any, Any], ...]
    ) -> None:
        seen_application_ids = set()

        for application in behavior_applications:
            if not isinstance(application, BehaviorApplication):
                raise TypeError(
                    "Each behavior application must implement the BehaviorApplication protocol; "
                    f"got {type(application).__name__!r}."
                )

            application_id = application.application_id
            if application_id in seen_application_ids:
                raise ValueError(f"Duplicate behavior application ID detected: {application_id!r}.")

            seen_application_ids.add(application_id)

    def __attach_behavior_applications(self) -> None:
        attached_applications = []

        try:
            for application in self.behavior_applications:
                application.on_attach(self.rid)
                attached_applications.append(application)
        except Exception:
            for application in reversed(attached_applications):
                try:
                    application.on_detach()
                except Exception:
                    logger.exception(
                        "Failed to detach behavior application %r while rolling back RSU %r attachment.",
                        application.application_id,
                        self.rid,
                    )
            raise

    def __detach_behavior_applications(self) -> None:
        first_exception = None

        for application in reversed(self.behavior_applications):
            try:
                application.on_detach()
            except Exception as exc:
                logger.exception(
                    "Failed to detach behavior application %r from RSU %r.",
                    application.application_id,
                    self.rid,
                )
                if first_exception is None:
                    first_exception = exc

        if first_exception is not None:
            raise first_exception

    def __validate_behavior_application_messages(self, messages: list[Any]) -> None:
        for message in messages:
            if not is_dataclass(message) or isinstance(message, type):
                raise TypeError(
                    "Behavior application input must be a list of dataclass instances; "
                    f"got {type(message).__name__!r}."
                )

            application_id = getattr(message, "application_id", None)
            if not isinstance(application_id, str) or not application_id:
                raise TypeError(
                    "Each behavior application message must define a non-empty 'application_id' attribute; "
                    f"got {type(message).__name__!r}."
                )

            if application_id not in self._behavior_applications_by_id:
                raise ValueError(
                    f"Behavior application message references unknown application_id {application_id!r}."
                )

    def __group_behavior_application_messages(self, messages: list[Any]) -> Dict[str, list[Any]]:
        grouped_messages = {application.application_id: [] for application in self.behavior_applications}

        for message in messages:
            grouped_messages[message.application_id].append(message)

        return grouped_messages

    def update_behavior_applications(self, messages: list[Any]) -> None:
        self.__validate_behavior_application_messages(messages)
        grouped_messages = self.__group_behavior_application_messages(messages)
        self.behavior_application_results = {}

        for application in self.behavior_applications:
            application_messages = grouped_messages[application.application_id]
            self.behavior_application_results[application.application_id] = application.process(application_messages)

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
        Run behavior applications for the provided message batch and
        execute the current RSU step side effects.
        """
        if messages is not None:
            self.update_behavior_applications(messages)

        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager, self.localizer, None)

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        try:
            self.__detach_behavior_applications()
        finally:
            self.perception_manager.destroy()
            self.localizer.destroy()
