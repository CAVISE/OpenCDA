# -*- coding: utf-8 -*-
"""
Basic class for RSU(Roadside Unit) management.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import carla

# CAVISE
import opencda.core.common.communication.serialize as cavise

from opencda.core.common.data_dumper import DataDumper
from opencda.core.sensing.perception.perception_manager import \
    PerceptionManager
from opencda.core.sensing.localization.rsu_localization_manager import \
    LocalizationManager


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

    current_id: int = -1

    def __init__(
            self,
            carla_world,
            config_yaml,
            carla_map,
            cav_world,
            current_time='',
            data_dumping=False):

        if 'id' in config_yaml:
            self.rid = -abs(int(config_yaml['id']))
            # The id of rsu is always a negative int
        else:
            self.rid = RSUManager.current_id
            RSUManager.current_id -= 1

        # read map from the world everytime is time-consuming, so we need
        # explicitly extract here
        self.carla_map = carla_map

        # cavise data payload
        self.rsu_data = {}

        # retrieve the configure for different modules
        # todo: add v2x module to rsu later
        sensing_config = config_yaml['sensing']
        sensing_config['localization']['global_position'] = config_yaml['spawn_position']
        sensing_config['perception']['global_position'] = config_yaml['spawn_position']

        # localization module
        self.localizer = LocalizationManager(carla_world,
                                             sensing_config['localization'],
                                             self.carla_map)

        # perception module
        self.perception_manager = PerceptionManager(vehicle=None,
                                                    config_yaml=sensing_config['perception'],
                                                    cav_world=cav_world,
                                                    carla_world=carla_world,
                                                    data_dump=data_dumping,
                                                    infra_id=self.rid)
        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager,
                                          self.rid,
                                          save_time=current_time)
        else:
            self.data_dumper = None

        cav_world.update_rsu_manager(self)

    def update_info(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # object detection todo: pass it to other CAVs for V2X percetion
        objects = self.perception_manager.detect(ego_pos)

        self.rsu_data['vid'] = str(self.rid)
        self.rsu_data['ego_spd'] = ego_spd
        self.rsu_data['ego_pos'] = cavise.SerializableTransform(ego_pos).to_dict()
        self.rsu_data['blue_vehicles'] = {}
        self.rsu_data['vehicles'] = []
        self.rsu_data['traffic_lights'] = []
        self.rsu_data['static_objects'] = [] # пока не используется
        self.rsu_data['from_who_received'] = [] # пока не используется

    def update_info_v2x(self, cav_list=[]):

        if cav_list != []:
            for cav_number_n_info in cav_list:
                self.rsu_data['blue_vehicles'][cav_number_n_info['vid']] = \
                {
                    'ego_spd' : cav_number_n_info['ego_spd'],
                    'ego_pos' : cav_number_n_info['ego_pos']
                }
                for blue_cav in cav_number_n_info['blue_vehicles']:
                    blue_vid, blue_info = blue_cav.items()
                    self.rsu_data['blue_vehicles'][blue_vid] = \
                    {
                        'ego_spd' : blue_info['ego_spd'],
                        'ego_pos' : blue_info['ego_pos']
                    }
                    
                tf = self.rsu_data['traffic_lights'] + cav_number_n_info['traffic_lights']
                tf_strings = [json.dumps(item, sort_keys=True) for item in tf]
                unique_tf_strings = set(tf_strings)
                unique_tf = [json.loads(item) for item in unique_tf_strings]
                self.rsu_data['traffic_lights'] = unique_tf

                veh = self.rsu_data['vehicles'] + cav_number_n_info['vehicles']
                veh_strings = [json.dumps(item, sort_keys=True) for item in veh]
                unique_veh_strings = set(veh_strings)
                unique_veh = [json.loads(item) for item in unique_veh_strings]
                self.rsu_data['vehicles'] = unique_veh
                
            #print(self.rsu_data, "result")


    def run_step(self):
        """
        Currently only used for dumping data.
        """
        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager,
                                      self.localizer,
                                      None)

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        self.perception_manager.destroy()
        self.localizer.destroy()
