import os
import zmq
import sys
import carla
import sumolib
import logging
import argparse
import omegaconf

from pathlib import Path
from typing import List, Union, Any
from dataclasses import dataclass

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api

from opencda.core.common.cav_world import CavWorld
from opencda.core.common.vehicle_manager import VehicleManager
from opencda.core.common.rsu_manager import RSUManager
from opencda.core.common.communication.serialize import MessageHandler
from opencda.core.application.platooning.platooning_manager import PlatooningManager
from opencda.core.common.codriving_model_manager import CodrivingModelManager


from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time, save_yaml

logger = logging.getLogger('cavise.scenario')


@dataclass
class Scenario:
    eval_manager: EvaluationManager
    scenario_manager: Union[sim_api.ScenarioManager, sim_api.CoScenarioManager]
    single_cav_list: List[VehicleManager]
    rsu_list: List[RSUManager]
    # TODO: find spectator type
    spectator: Any
    cav_world: CavWorld
    codriving_model_manager: CodrivingModelManager # [CoDrivingInt]
    platoon_list: List[PlatooningManager]
    # TODO: find bg cars type
    bg_veh_list: Any
    scenario_name: str

    def __init__(self, opt: argparse.Namespace, scenario_params: omegaconf.OmegaConf):
        self.node_ids = {'cav': [],
                         'rsu': [],
                         'platoon': []}
        self.scenario_name = opt.test_scenario
        self.scenario_params, current_time = add_current_time(scenario_params)
        logger.info(f'running scenario with name: {self.scenario_name}; current time: {current_time}')

        self.cav_world = CavWorld(opt.apply_ml, opt.with_capi)
        logger.info(f'created cav world, using apply_ml = {opt.apply_ml}, with_capi = {opt.with_capi}')

        xodr_path = None
        if opt.xodr:
            xodr_path = Path('opencda/sumo-assets') / self.scenario_name / f'{self.scenario_name}.xodr'
            logger.info(f'loading xodr map with name: {xodr_path}')

        town = None
        if xodr_path is None:
            if 'town' not in scenario_params['world']:
                logger.error(f'You must specify xodr parameter or town key in opencda/scenario_testing/config_yaml/{self.scenario_name}.yaml')
                sys.exit(1)
            town = scenario_params['world']['town']
            logger.info(f'using town: {town}')

        if opt.cosim:
            sumo_cfg = Path('opencda/sumo-assets') / self.scenario_name
            self.scenario_manager = sim_api.CoScenarioManager(
                scenario_params=scenario_params,
                apply_ml=opt.apply_ml,
                carla_version=opt.version,
                town=town,
                cav_world=self.cav_world,
                sumo_file_parent_path=sumo_cfg,
                node_ids=self.node_ids
            )
        else:
            self.scenario_manager = sim_api.ScenarioManager(
                scenario_params=scenario_params,
                apply_ml=opt.apply_ml,
                carla_version=opt.version,
                xodr_path=xodr_path,
                town=town,
                cav_world=self.cav_world
            )

        if self.cav_world.comms_manager is not None:
            self.cav_world.comms_manager.create_socket(zmq.PAIR, 'connect')
            self.message_handler = MessageHandler()
            logger.info('running: creating message handler')
        else:
            self.message_handler = None

        logger.info(f'using scenario manager of type: {type(self.scenario_manager)}')

        data_dump = opt.with_coperception and opt.model_dir is not None or opt.record
        logger.info('data dump is ' + ('ON' if data_dump else 'OFF'))

        if data_dump:
            self.scenario_manager.client.start_recorder(f'{self.scenario_name}.log', True)

            save_yaml_name = Path('simulation_output/data_dumping') / current_time / 'data_protocol.yaml'
            logger.info(f'saving params to {save_yaml_name}')
            os.makedirs(os.path.dirname(save_yaml_name), exist_ok=True)
            save_yaml(scenario_params, save_yaml_name)

            if opt.with_coperception and opt.model_dir:
                from opencda.core.common.coperception_model_manager import CoperceptionModelManager
                if opt.fusion_method not in ['late', 'early', 'intermediate']:
                    logger.error('Invalid fusion method: must be one of "late", "early", or "intermediate".')
                    sys.exit(1)

                if not os.path.isdir(opt.model_dir):
                    logger.error(f'Model directory "{opt.model_dir}" does not exist.')
                    sys.exit(1)

                self.coperception_model_manager = CoperceptionModelManager(opt=opt, current_time=current_time, message_handler=self.message_handler)
                logger.info('created cooperception manager')

        elif opt.record:
            logger.info('beginning to record the simulation')
            self.scenario_manager.client.start_recorder(f'{self.scenario_name}.log', True)
        else:
            self.coperception_model_manager = None

        # [CoDrivingInt]
        if opt.with_mtp:
            logger.info('Codriving Model is initialized')

            net = sumolib.net.readNet(f"opencda/sumo-assets/{self.scenario_name}/{self.scenario_name}.net.xml")
            nodes = net.getNodes()

            # TODO: Replace with params from scenario file
            self.codriving_model_manager = CodrivingModelManager(
                pretrained='opencda/codriving_models/gnn_mtl_gnn/model_rot_gnn_mtl_np_sumo_0911_e3_1930.pth',
                model_name="GNN_mtl_gnn",
                nodes=nodes
            )
        # [CoDrivingInt]

        self.platoon_list, self.node_ids['platoon'] = self.scenario_manager.create_platoon_manager(
            map_helper=map_api.spawn_helper_2lanefree, data_dump=data_dump
        )
        logger.info(f'created platoon list of size {len(self.platoon_list)}')

        self.single_cav_list, self.node_ids['cav'] = self.scenario_manager.create_vehicle_manager(
            application=['single'], map_helper=map_api.spawn_helper_2lanefree, data_dump=data_dump
        )
        logger.info(f'created single cavs of size {len(self.single_cav_list)}')

        _, self.bg_veh_list = self.scenario_manager.create_traffic_carla()
        logger.info(f'created background traffic of size {len(self.bg_veh_list)}')

        self.rsu_list, self.node_ids['rsu'] = self.scenario_manager.create_rsu_manager(data_dump=data_dump)
        logger.info(f'created RSU list of size {len(self.rsu_list)}')

        self.eval_manager = EvaluationManager(
            self.scenario_manager.cav_world,
            script_name=self.scenario_name,
            current_time=scenario_params['current_time']
        )

        # [CoDrivingInt]
        if opt.with_mtp:
            logger.info('Codriving Model is initialized')

            net = sumolib.net.readNet(f"opencda/sumo-assets/{self.scenario_name}/{self.scenario_name}.net.xml")
            nodes = net.getNodes()

            # TODO: Replace with params from scenario file
            self.codriving_model_manager = CodrivingModelManager(
                pretrained='opencda/codriving_models/gnn_mtl_gnn/model_rot_gnn_mtl_np_sumo_0911_e3_1930.pth', \
                model_name="GNN_mtl_gnn",
                nodes=nodes,
                excluded_nodes=None, #scenario_params['excluded_nodes'] if scenario_params['excluded_nodes'] else None
            )
        # [CoDrivingInt]

        self.spectator = self.scenario_manager.world.get_spectator()

    def run(self, opt: argparse.Namespace):
        if self.cav_world.comms_manager is not None:
            self.cav_world.comms_manager.create_socket(zmq.PAIR, 'connect')
            message_handler = MessageHandler()
            logger.info('running: creating message handler')
        else:
            message_handler = None

        tick_number = 0
        if self.coperception_model_manager is not None:
            from opencda.core.common.coperception_model_manager import DirectoryProcessor
            now_directory = 'simulation_output/data_dumping/sample/now'
            directory_processor = DirectoryProcessor(source_directory='simulation_output/data_dumping', now_directory=now_directory)
            os.makedirs(now_directory, exist_ok=True)
            directory_processor.clear_directory_now()

        while True:
            # [CoDrivingInt]
            if opt.with_mtp:
                self.codriving_model_manager.make_trajs(
                    carla_vmanagers=self.single_cav_list
                )
            # [CoDrivingInt]

            tick_number += 1
            if opt.ticks and tick_number > opt.ticks:
                break
            logger.debug(f'running: simulation tick: {tick_number}')
            self.scenario_manager.tick()

            if not opt.free_spectator and any(array is not None for array in [self.single_cav_list, self.platoon_list]):
                if len(self.single_cav_list) > 0:
                    transform = self.single_cav_list[0].vehicle.get_transform()
                    self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
                else:
                    transform = self.platoon_list[0].vehicle_manager_list[0].vehicle.get_transform()
                    self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            if self.platoon_list is not None:
                logger.debug('updating platoons')
                for platoon in self.platoon_list:
                    platoon.update_information()
                    platoon.run_step()

            if self.single_cav_list is not None:
                logger.debug('updating single cavs')
                for single_cav in self.single_cav_list:
                    single_cav.update_info()
                    control = single_cav.run_step()
                    single_cav.vehicle.apply_control(control)

            if self.rsu_list is not None:
                logger.debug('updating RSUs')
                for rsu in self.rsu_list:
                    rsu.update_info()
                    rsu.run_step()

            if self.coperception_model_manager is not None:
                try:
                    logger.info(f'Processing {tick_number} tick')
                    directory_processor.clear_directory_now()
                    directory_processor.process_directory(tick_number)
                    logger.info(f'Successfully processed {tick_number} tick')
                except Exception as e:
                    logger.warning(f'An error occurred during proceesing {tick_number} tick: {e}')

                self.coperception_model_manager.make_dataset()

                if self.cav_world.comms_manager is not None:  # TODO: Надо добавить возможность Artery работать без OpenCOOD
                    self.coperception_model_manager.opencood_dataset.get_entity_item(idx=0)  # TODO: Надо разобраться с тем, как выбирать ego в моделях совместного восприятия
                    msg = self.message_handler.serialize_to_string()
                    self.cav_world.comms_manager.send_message(msg)
                    logger.info(f'{round(len(msg) / (1 << 20), 3)} MB about to be sent')

                    msg = self.cav_world.comms_manager.receive_message()
                    logger.info(f'{round(len(msg) / (1 << 20), 3)} MB were received')
                    self.message_handler.deserialize_from_string(msg)

                self.coperception_model_manager.make_prediction(tick_number)

                if self.cav_world.comms_manager is not None:
                    self.message_handler.clear_messages()

    def finalize(self, opt: argparse.Namespace):
        if opt.record:
            self.scenario_manager.client.stop_recorder()
            logger.info('finalizing: stopping recorder')

        if self.eval_manager is not None:
            self.eval_manager.evaluate()
            logger.info('finalizing: evaluating results')

        if self.coperception_model_manager is not None:
            self.coperception_model_manager.final_eval()

        if self.single_cav_list is not None:
            logger.info(f'finalizing: destroying {len(self.single_cav_list)} single cavs')
            for v in self.single_cav_list:
                v.destroy()

        if self.rsu_list is not None:
            logger.info(f'finalizing: destroying {len(self.rsu_list)} RSUs')
            for r in self.rsu_list:
                r.destroy()

        if self.scenario_manager is not None:
            self.scenario_manager.close()
            logger.info('finalizing: evaluating results')

        if self.platoon_list is not None:
            logger.info(f'finalizing: destroying {len(self.platoon_list)} platoons')
            for platoon in self.platoon_list:
                platoon.destroy()

        if self.bg_veh_list is not None:
            logger.info(f'finalizing: destroying {len(self.bg_veh_list)} background cars')
            for v in self.bg_veh_list:
                v.destroy()


def run_scenario(opt, scenario_params) -> None:
    raised_error = scenario = None
    try:
        scenario = Scenario(opt, scenario_params)
        scenario.run(opt)
    except Exception as error:
        raised_error = error
    finally:
        logger.info("Wrapping things up... Please don't press Ctrl+C")
        if scenario:
            scenario.finalize(opt)
        if raised_error is not None:
            raise raised_error
