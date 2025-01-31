#-*- coding: utf-8 -*-
#Author: Runsheng Xu [rxx3386@ucla.edu](mailto:rxx3386@ucla.edu)
#License: TDG-Attribution-NonCommercial-NoDistrib

import os
import zmq
import time
import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api

from opencda.core.common.cav_world import CavWorld
from opencda.core.common.communication.serialize import MessageHandler
from opencda.core.common.coperception_model_manager import CoperceptionModelManager, DirectoryProcessor

import opencda.core.common.communication.protos.cavise.artery_pb2 as proto_artery

from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time, save_yaml

from google.protobuf.json_format import MessageToJson


eval_manager = None
scenario_manager = None
single_cav_list = None
rsu_list = None
spectator = None
cav_world = None
scenario_name = "1551_carla_cavs"

OPENCDA_MESSAGE_LOCATION = os.environ.get('CAVISE_ROOT_DIR') + '/simdata/message_opencda.json'
ARTERY_MESSAGE_LOCATION = os.environ.get('CAVISE_ROOT_DIR') + '/simdata/message_artery.json'


def init(opt, scenario_params) -> None:
    global eval_manager, scenario_manager, single_cav_list, spectator, cav_world, rsu_list, coperception_model_manager

    scenario_params = add_current_time(scenario_params)
    cav_world = CavWorld(opt.apply_ml, opt.with_capi)

    cavise_root = os.environ.get('CAVISE_ROOT_DIR')
    if not cavise_root:
        raise EnvironmentError('missing cavise root!')
    
    sumo_cfg = f'{cavise_root}/opencda/opencda/assets/{scenario_name}'
    scenario_manager = sim_api.CoScenarioManager(
        scenario_params,
        opt.apply_ml,
        opt.version,
        town='Town06',
        cav_world=cav_world,
        sumo_file_parent_path=sumo_cfg,
        with_capi=opt.with_capi
    )

    data_dump = opt.with_coperception and (opt.model_dir is not None) or opt.record
    coperception_model_manager = None

    if data_dump:
        scenario_manager.client.start_recorder(f"{scenario_name}.log", True)

        current_path = os.path.dirname(os.path.realpath(__file__))
        save_yaml_name = os.path.join(
            current_path,
            '../../data_dumping',
            scenario_params['current_time'],
            'data_protocol.yaml'
        )
        os.makedirs(os.path.dirname(save_yaml_name), exist_ok=True)
        save_yaml(scenario_params, save_yaml_name)

        if opt.with_coperception and opt.model_dir:
            coperception_model_manager = CoperceptionModelManager(opt=opt)


    single_cav_list = scenario_manager.create_vehicle_manager(
        application=['single'],
		map_helper=map_api.spawn_helper_2lanefree,
        data_dump=data_dump
    )

    rsu_list = \
        scenario_manager.create_rsu_manager(data_dump=data_dump)

    eval_manager = EvaluationManager(
        scenario_manager.cav_world, 
        script_name=scenario_name, 
        current_time=scenario_params['current_time']
    )

    spectator = scenario_manager.world.get_spectator()


def finalize(opt) -> None:
    global eval_manager, scenario_manager, single_cav_list, rsu_list

    if eval_manager is not None:
        eval_manager.evaluate()
    if scenario_manager is not None:
        scenario_manager.close()
    if single_cav_list is not None:
        for v in single_cav_list:
            v.destroy()
    if rsu_list is not None:
        for r in rsu_list:
            r.destroy()


def run() -> None:
    global eval_manager, scenario_manager, single_cav_list, spectator, rsu_list, coperception_model_manager

    # this sim requires communication manager
    if cav_world.comms_manager is not None:
        cav_world.comms_manager.create_socket(zmq.PAIR, 'connect')
        message_handler = MessageHandler()
    else:
        message_handler = None

    tick_number = -1
    dir_number = -1
    directory_processor = DirectoryProcessor(source_directory="data_dumping",
                                             now_directory="data_dumping/sample/now")
    os.makedirs("data_dumping/sample/now", exist_ok=True)
    directory_processor.clear_directory_now()

    spectator.set_transform(carla.Transform(carla.Location(x=155.774963, y=150.137070, z=142.591293), carla.Rotation(pitch=-90, yaw=-90, roll=0)))

    while True:
        scenario_manager.tick()
        tick_number += 1
        transform = single_cav_list[2].vehicle.get_transform()
        #spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        for single_cav in single_cav_list:
            single_cav.update_info()
            if message_handler is not None:
                message_handler.set_cav_data(single_cav.cav_data)

        for rsu in rsu_list:
            rsu.update_info()
            if message_handler is not None:
                message_handler.set_cav_data(rsu.rsu_data)

        if coperception_model_manager and tick_number >= 60:
            try:
                directory_processor.clear_directory_now()
                directory_processor.process_directory(tick_number)
                dir_number = tick_number
            except Exception as e:
                if tick_number % 2 == 0:
                    logger.warning(f"[WARNING] An error occurred: {e}")

        if cav_world.comms_manager is not None:
        
            # be verbose!
            json_output = MessageToJson(message_handler.opencda_message, preserving_proto_field_name=True)
            with open(OPENCDA_MESSAGE_LOCATION, 'w') as json_file:
                json_file.write(json_output)
            
            out_message = message_handler.serialize_to_string()
            cav_world.comms_manager.send_message(out_message)
            in_message = cav_world.comms_manager.receive_message()
            v2x_info = MessageHandler.deserialize_from_string(in_message)

            # be verbose!
            parsed = proto_artery.Artery_message()
            parsed.ParseFromString(in_message)
            json_output = MessageToJson(parsed, preserving_proto_field_name=True)
            with open(ARTERY_MESSAGE_LOCATION, 'w') as json_file:
                json_file.write(json_output)
        else:
            v2x_info = {}
            
        for single_cav in single_cav_list:
            cav_list = []
            if len(v2x_info) > 0:
                cav_list = v2x_info[str(single_cav.vid)]['cav_list']
            # else:
            #     print('Data has been lost!')
            single_cav.update_info_v2x(cav_list=cav_list)
            control = single_cav.run_step()
            single_cav.vehicle.apply_control(control)

        for rsu in rsu_list:
            cav_list = []
            if len(v2x_info) > 0:
                cav_list = v2x_info[str(rsu.rid)]['cav_list']
            # else:
            #     print('Data has been lost!')
            rsu.update_info_v2x(cav_list=cav_list)
            rsu.run_step()

        if coperception_model_manager and coperception_model_manager.opt.fusion_method and coperception_model_manager.opt.model_dir and dir_number == tick_number:
            coperception_model_manager.make_pred()


def run_scenario(opt, scenario_params) -> None:
    global eval_manager, scenario_manager, single_cav_list, spectator

    raised_error = None
    try:
        init(opt, scenario_params)
        run()
    except Exception as error:
        raised_error = error
    finally:
        finalize(opt)
        if raised_error is not None:
            raise raised_error