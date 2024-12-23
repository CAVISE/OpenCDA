#-*- coding: utf-8 -*-
#Author: Runsheng Xu [rxx3386@ucla.edu](mailto:rxx3386@ucla.edu)
#License: TDG-Attribution-NonCommercial-NoDistrib

import os
import zmq
import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api

from opencda.core.common.cav_world import CavWorld
from opencda.core.common.communication.serialize import MessageHandler

import opencda.core.common.communication.protos.cavise.artery_pb2 as proto_artery

from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time

from google.protobuf.json_format import MessageToJson


eval_manager = None
scenario_manager = None
single_cav_list = None
spectator = None
cav_world = None

OPENCDA_MESSAGE_LOCATION = os.environ.get('CAVISE_ROOT_DIR') + '/simdata/opencda/message.json'
ARTERY_MESSAGE_LOCATION = os.environ.get('CAVISE_ROOT_DIR') + '/simdata/artery/message.json'


def init(opt, scenario_params) -> None:
    global eval_manager, scenario_manager, single_cav_list, spectator, cav_world

    scenario_params = add_current_time(scenario_params)
    cav_world = CavWorld(opt.apply_ml, opt.with_cccp)

    cavise_root = os.environ.get('CAVISE_ROOT_DIR')
    if not cavise_root:
        raise EnvironmentError('missing cavise root!')
    
    sumo_cfg = f'{cavise_root}/opencda/opencda/scenario_testing/config_sumo/realistic_town06_cosim'
    scenario_manager = sim_api.CoScenarioManager(
        scenario_params,
        opt.apply_ml,
        opt.version,
        town='Town06',
        cav_world=cav_world,
        sumo_file_parent_path=sumo_cfg
    )

    single_cav_list = scenario_manager.create_vehicle_manager(
        application=['single'],
		map_helper=map_api.
		spawn_helper_2lanefree
    )

    eval_manager = EvaluationManager(
        scenario_manager.cav_world, 
        script_name='realistic_town06_cosim', 
        current_time=scenario_params['current_time']
    )

    spectator = scenario_manager.world.get_spectator()


def finalize() -> None:
    global eval_manager, scenario_manager, single_cav_list

    if eval_manager is not None:
        eval_manager.evaluate()
    if scenario_manager is not None:
        scenario_manager.close()
    if single_cav_list is not None:
        for v in single_cav_list:
            v.destroy()


def run() -> None:
    global eval_manager, scenario_manager, single_cav_list, spectator

    # this sim requires communication manager
    if cav_world.comms_manager is None:
        raise AttributeError("This sim requires communication manager")
    cav_world.comms_manager.create_socket(zmq.PAIR, 'connect')

    message_handler = MessageHandler()
    while True:
        scenario_manager.tick()
        transform = single_cav_list[0].vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        for _, single_cav in enumerate(single_cav_list):
            single_cav.update_info()
            message_handler.set_cav_data(single_cav.cav_data)
        
        # be verbose!
        json_output = MessageToJson(message_handler.opencda_message, including_default_value_fields=True, preserving_proto_field_name=True)
        with open(OPENCDA_MESSAGE_LOCATION, 'w') as json_file:
            json_file.write(json_output)
        
        out_message = message_handler.serialize_to_string()
        cav_world.comms_manager.send_message(out_message)
        in_message = cav_world.comms_manager.receive_message()
        v2x_info = MessageHandler.deserialize_from_string(in_message)

        # be verbose!
        parsed = proto_artery.Artery_message()
        parsed.ParseFromString(in_message)
        json_output = MessageToJson(parsed, including_default_value_fields=True, preserving_proto_field_name=True)
        with open(ARTERY_MESSAGE_LOCATION, 'w') as json_file:
            json_file.write(json_output)
        
        for _, single_cav in enumerate(single_cav_list):
            cav_list = []
            if len(v2x_info) > 0:
                cav_list = v2x_info[str(int(single_cav.vid.replace('-', ''), 16))]['cav_list']
            else:
                print('Data has been lost!')
            single_cav.update_info_v2x(cav_list=cav_list)
            control = single_cav.run_step()
            single_cav.vehicle.apply_control(control)


def run_scenario(opt, scenario_params) -> None:
    global eval_manager, scenario_manager, single_cav_list, spectator

    raised_error = None
    try:
        init(opt, scenario_params)
        run()
    except Exception as error:
        raised_error = error
    finally:
        finalize()
        if raised_error is not None:
            raise raised_error

