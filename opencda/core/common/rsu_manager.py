# -*- coding: utf-8 -*-
"""
Basic class for RSU(Roadside Unit) management.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import logging
from typing import Any, Set

# CAVISE

from opencda.core.common.data_dumper import DataDumper
from opencda.core.sensing.perception.perception_manager import PerceptionManager
from opencda.core.sensing.localization.rsu_localization_manager import LocalizationManager

logger = logging.getLogger("cavise.rsu_manager")


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
    used_ids: Set[int] = set()

    def __init__(
        self, carla_world, config_yaml, carla_map, cav_world, current_time="", data_dumping=False, autogenerate_id_on_failure=True
    ):  # TODO: Привязать к конфигу сценария
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

    def run_step(self):
        """
        Currently only used for dumping data.
        """
        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager, self.localizer, None)

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        self.perception_manager.destroy()
        self.localizer.destroy()
