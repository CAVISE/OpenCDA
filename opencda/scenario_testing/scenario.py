from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import carla
from omegaconf import DictConfig, OmegaConf

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.application.platooning.platooning_manager import PlatooningManager
from opencda.core.attack.adversary_framework import Attack, AttackManager, AttackResult, AttackSpec
from opencda.core.common.cav_world import CavWorld
from opencda.core.common.coperception_data_processor import CoperceptionDataProcessor
from opencda.core.common.rsu_manager import RSUManager
from opencda.core.common.vehicle_manager import VehicleManager
from opencda.core.sensing.perception.perception_manager import PerceptionRequirements
from opencda.scenario_testing.types import NodeSnapshot, SimulationSnapshot
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import YamlDict, add_current_time, load_yaml, save_yaml

from opencda.core.application.behavior import TransportMessage

if TYPE_CHECKING:
    from opencda.core.common.coperception_model_manager import CoperceptionModelManager
    from opencda.core.common.communication.communication_manager import CommunicationManager
    from opencda.core.common.communication.payload_handler import PayloadHandler

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.scenario")


@dataclass
class Scenario:
    eval_manager: EvaluationManager
    scenario_manager: sim_api.ScenarioManager | sim_api.CoScenarioManager
    single_cav_list: list[VehicleManager]
    rsu_list: list[RSUManager]
    spectator: carla.Actor
    cav_world: CavWorld
    platoon_list: list[PlatooningManager]
    bg_veh_list: list[carla.Actor]
    scenario_name: str

    def _abort_simulation(self, message: str) -> NoReturn:
        logger.error(message)
        raise RuntimeError(message)

    def _require_coperception_data_processor(self) -> CoperceptionDataProcessor:
        if self.coperception_data_processor is None:
            self._abort_simulation("Coperception data processor is required, but it was not initialized.")
        return self.coperception_data_processor

    def _require_cav_world(self) -> CavWorld:
        cav_world = self.scenario_manager.cav_world
        if cav_world is None:
            self._abort_simulation("Scenario manager was initialized without CavWorld; simulation cannot continue.")
        return cav_world

    def __init__(self, opt: argparse.Namespace, scenario_params: DictConfig) -> None:
        self.node_ids: dict[str, dict[int, str]] = {"cav": {}, "rsu": {}, "platoon": {}}
        self.scenario_name = opt.test_scenario
        scenario_config = cast(YamlDict, OmegaConf.to_container(scenario_params, resolve=True))
        self.scenario_params, current_time = add_current_time(scenario_config)
        scenario_config = self.scenario_params
        logger.info(f"running scenario with name: {self.scenario_name}; current time: {current_time}")

        self.cav_world = CavWorld(opt.apply_ml)
        logger.info(f"created cav world, using apply_ml = {opt.apply_ml}")

        self.payload_handler: PayloadHandler | None = None
        self.communication_manager: CommunicationManager | None = None
        self.coperception_model_manager: CoperceptionModelManager | None = None
        self.coperception_data_processor: CoperceptionDataProcessor | None = None
        cp_vis_config = None

        xodr_path: str | None = None
        if opt.xodr:
            xodr_path = str(Path("opencda/sumo-assets") / self.scenario_name / f"{self.scenario_name}.xodr")
            logger.info(f"loading xodr map with name: {xodr_path}")

        town: str | None = None
        if xodr_path is None:
            if "town" not in scenario_config["world"]:
                self._abort_simulation(
                    f"You must specify xodr parameter or town key in opencda/scenario_testing/config_yaml/{self.scenario_name}.yaml"
                )
            town = cast(str, scenario_config["world"]["town"])
            logger.info(f"using town: {town}")

        if opt.cosim:
            sumo_cfg = str(Path("opencda/sumo-assets") / self.scenario_name)
            self.scenario_manager = sim_api.CoScenarioManager(
                scenario_params=scenario_config,
                apply_ml=opt.apply_ml,
                carla_version=opt.version,
                town=town,
                cav_world=self.cav_world,
                sumo_file_parent_path=sumo_cfg,
                node_ids=self.node_ids,
                carla_host=opt.carla_host,
                carla_timeout=opt.carla_timeout,
            )
        else:
            self.scenario_manager = sim_api.ScenarioManager(
                scenario_params=scenario_config,
                apply_ml=opt.apply_ml,
                carla_version=opt.version,
                xodr_path=xodr_path,
                town=town,
                cav_world=self.cav_world,
                carla_host=opt.carla_host,
                carla_timeout=opt.carla_timeout,
            )
        self.cav_world = self._require_cav_world()

        if opt.with_capi:
            from opencda.core.common.communication import toolchain

            toolchain.CommunicationToolchain.handle_messages(["entity", "opencda", "artery", "capi"])
            from opencda.core.common.communication.communication_manager import CommunicationManager
            from opencda.core.common.communication.payload_handler import PayloadHandler

            self.communication_manager = CommunicationManager(
                artery_address=f"tcp://{opt.artery_host}",
                artery_send_timeout=opt.artery_send_timeout,
                artery_receive_timeout=opt.artery_receive_timeout,
            )
            self.payload_handler = PayloadHandler()
            logger.info("running: creating message handler")

        logger.info(f"using scenario manager of type: {type(self.scenario_manager)}")
        logger.info(f"data dump is {'ON' if opt.record else 'OFF'}")

        if opt.record:
            logger.info("beginning to record the simulation in simulation_output/data_dumping")
            self.scenario_manager.client.start_recorder(f"{self.scenario_name}.log", True)

            save_yaml_name = Path("simulation_output/data_dumping") / current_time / "data_protocol.yaml"
            logger.info(f"saving params to {save_yaml_name}")
            os.makedirs(os.path.dirname(save_yaml_name), exist_ok=True)
            save_yaml(scenario_config, save_yaml_name)

        perception_requirements = PerceptionRequirements.from_runtime_flags(
            data_dump=opt.record,
            with_coperception=opt.with_coperception,
        )

        self.platoon_list, platoon_node_ids = self.scenario_manager.create_platoon_manager(
            map_helper=map_api.spawn_helper_2lanefree,
            perception_requirements=perception_requirements,
        )
        self.node_ids["platoon"] = cast(dict[int, str], platoon_node_ids)
        logger.info(f"created platoon list of size {len(self.platoon_list)}")

        self.single_cav_list, cav_node_ids = self.scenario_manager.create_vehicle_manager(
            application=["single"],
            map_helper=map_api.spawn_helper_2lanefree,
            perception_requirements=perception_requirements,
        )
        self.node_ids["cav"] = cast(dict[int, str], cav_node_ids)
        logger.info(f"created single cavs of size {len(self.single_cav_list)}")

        _, self.bg_veh_list = self.scenario_manager.create_traffic_carla()
        logger.info(f"created background traffic of size {len(self.bg_veh_list)}")

        self.rsu_list, rsu_node_ids = self.scenario_manager.create_rsu_manager(
            perception_requirements=perception_requirements,
        )
        self.node_ids["rsu"] = cast(dict[int, str], rsu_node_ids)
        logger.info(f"created RSU list of size {len(self.rsu_list)}")

        if opt.with_coperception and opt.model_dir:
            if not os.path.isdir(opt.model_dir):
                self._abort_simulation(f'Model directory "{opt.model_dir}" does not exist; cannot initialize cooperative perception manager.')

            cp_vis_config = OmegaConf.to_container(
                scenario_params.get("cooperative_perception_visualization", {}),
                resolve=True,
            )

            CoperceptionManagerClass: type[CoperceptionModelManager]
            if opt.with_advcp:
                from opencda.core.attack.advcp.adv_coperception_model_manager import AdvCoperceptionModelManager as CoperceptionManagerClass
            else:
                from opencda.core.common.coperception_model_manager import CoperceptionModelManager as CoperceptionManagerClass

            self.coperception_model_manager = CoperceptionManagerClass(
                opt=opt,
                current_time=current_time,
                payload_handler=self.payload_handler,
                visualization_config=cp_vis_config,
            )
            valid_agent_ids = [vehicle_manager.id for vehicle_manager in self.single_cav_list]
            valid_agent_ids.extend(rsu_manager.id for rsu_manager in self.rsu_list)
            if hasattr(self.coperception_model_manager, "validate_advcp_agents"):
                advcp_ready = self.coperception_model_manager.validate_advcp_agents(valid_agent_ids)
                if not advcp_ready:
                    from opencda.core.common.coperception_model_manager import CoperceptionModelManager

                    logger.warning("AdvCP validation failed. Falling back to the default cooperative perception manager for this run.")
                    self.coperception_model_manager = CoperceptionModelManager(
                        opt=opt,
                        current_time=current_time,
                        payload_handler=self.payload_handler,
                        visualization_config=cp_vis_config,
                    )

            """
            TODO: Create decorators to write such stuff

            @cavise.SimObject
            class SomeCoolManager
            that would at least take the logging part - writing "creating SomeCoolManager manager", destroying SomeCoolManager manager"

            Also ideally it would also somehow verify manager configs, for example.
            """
            self.coperception_data_processor = CoperceptionDataProcessor()
            logger.info("created cooperception manager")

        self.scenario_manager.create_custom_actor_manager(application=["single"], map_helper=map_api.spawn_helper_2lanefree, data_dump=opt.record)
        logger.info("created single custom actors")

        cav_world = self._require_cav_world()
        self.eval_manager = EvaluationManager(cav_world, script_name=self.scenario_name, current_time=scenario_config["current_time"])

        self.spectator = self.scenario_manager.world.get_spectator()

        self.messages: list[TransportMessage[Any]] = []
        self.simulation_snapshot = SimulationSnapshot(tick=-1)
        self.attack_manager = AttackManager()
        attacks_config = scenario_config.get("attacks", [])
        if not isinstance(attacks_config, list):
            self._abort_simulation("Scenario config field 'attacks' must be a list of attack names.")

        attacks: list[Attack] = []
        for attack_name in attacks_config:
            if not isinstance(attack_name, str):
                self._abort_simulation("Scenario config field 'attacks' must contain only attack names as strings.")

            attack_config_path = Path("opencda/core/attack/adversary_framework/attacks") / attack_name / "config.yaml"
            attack_config = load_yaml(attack_config_path)
            attack_spec = AttackSpec.from_dict(cast(YamlDict, attack_config["attack"]))
            attacks.append(Attack.from_spec(attack_spec))

        self.attacks = attacks
        self.attack_results: tuple[AttackResult, ...] = ()

    def _build_simulation_snapshot(self, tick: int) -> SimulationSnapshot:
        vehicle_nodes = tuple(
            NodeSnapshot(
                node_id=vehicle_manager.id,
                node_type="vehicle",
                service_states=dict(vehicle_manager.behavior_service_states),
            )
            for vehicle_manager in chain(
                self.single_cav_list,
                *(platoon.vehicle_manager_list for platoon in self.platoon_list),
            )
        )

        rsu_nodes = tuple(
            NodeSnapshot(
                node_id=rsu.id,
                node_type="rsu",
                service_states=dict(rsu.behavior_service_states),
            )
            for rsu in self.rsu_list
        )

        return SimulationSnapshot(
            tick=tick,
            vehicle_nodes=vehicle_nodes,
            rsu_nodes=rsu_nodes,
        )

    def run(self, opt: argparse.Namespace) -> None:
        if self.communication_manager is None:
            self.default_loop(opt)
        else:
            self.capi_loop(opt)

    def default_loop(self, opt: argparse.Namespace) -> None:
        tick_number = -1
        while True:
            tick_number += 1

            if opt.ticks and tick_number > opt.ticks:
                break
            logger.debug(f"running: simulation tick: {tick_number}")
            self.scenario_manager.sumo_tick()
            self.scenario_manager.tick()

            if not opt.free_spectator and any(array is not None for array in [self.single_cav_list, self.platoon_list]):
                if len(self.single_cav_list) > 0:
                    transform = self.single_cav_list[0].vehicle.get_transform()
                    self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
                else:
                    transform = self.platoon_list[0].vehicle_manager_list[0].vehicle.get_transform()
                    self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            if self.platoon_list is not None:
                logger.debug("updating platoons")
                for platoon in self.platoon_list:
                    platoon.update_information()

            new_messages: list[TransportMessage[Any]] = []
            if self.single_cav_list is not None:
                logger.debug("updating single cavs")

                for single_cav in self.single_cav_list:
                    single_cav.update_info()

            if self.rsu_list is not None:
                logger.debug("updating RSUs")
                for rsu in self.rsu_list:
                    rsu.update_info()

            if self.coperception_model_manager is not None and tick_number > 0:
                logger.info(f"Processing {tick_number} tick")
                memory_structure = self._require_coperception_data_processor().build_live_memory(self.single_cav_list, self.rsu_list, tick_number)
                if memory_structure is None:
                    logger.warning(f"Live cooperative perception data for tick {tick_number} is not available.")
                else:
                    self.coperception_model_manager.update_dataset(memory_structure)
                    self.coperception_model_manager.make_prediction(tick_number)
                    logger.info(f"Successfully processed {tick_number} tick")

            if self.platoon_list is not None:
                for platoon in self.platoon_list:
                    platoon.run_step()

            if self.single_cav_list is not None:
                for single_cav in self.single_cav_list:
                    cav_messages, _ = single_cav.run_step(messages=self.messages)
                    new_messages.extend(cav_messages)

            if self.rsu_list is not None:
                for rsu in self.rsu_list:
                    rsu_messages, _ = rsu.run_step(messages=self.messages)
                    new_messages.extend(rsu_messages)

            self.messages = new_messages
            self.simulation_snapshot = self._build_simulation_snapshot(tick_number)

            self.attack_results = self.attack_manager.evaluate(
                self.attacks,
                self.simulation_snapshot,
                service_resolver=self._require_cav_world().resolve_behavior_service,
            )
            for result in self.attack_results:
                logger.info("attack=%s status=%s reason=%s", result.attack_name, result.status.value, result.reason)

    def capi_loop(self, opt: argparse.Namespace) -> None:
        if self.communication_manager is None:
            self._abort_simulation("CommunicationManager is required for CAPI flow, but it was not initialized.")
        if self.payload_handler is None:
            self._abort_simulation("PayloadHandler is required for CAPI flow, but it was not initialized.")
        communication_manager = self.communication_manager
        payload_handler = self.payload_handler
        tick_number = -1
        while True:
            tick_number += 1

            if opt.ticks and tick_number > opt.ticks:
                break
            logger.debug(f"running: simulation tick: {tick_number}")
            self.scenario_manager.tick()

            if not opt.free_spectator and any(array is not None for array in [self.single_cav_list, self.platoon_list]):
                if len(self.single_cav_list) > 0:
                    transform = self.single_cav_list[0].vehicle.get_transform()
                    self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
                else:
                    transform = self.platoon_list[0].vehicle_manager_list[0].vehicle.get_transform()
                    self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            # TODO: Add aim service support

            if self.platoon_list is not None:
                logger.debug("updating platoons")
                for platoon in self.platoon_list:
                    platoon.update_information()

            if self.single_cav_list is not None:
                logger.debug("updating single cavs")
                for single_cav in self.single_cav_list:
                    single_cav.update_info()

            if self.rsu_list is not None:
                logger.debug("updating RSUs")
                for rsu in self.rsu_list:
                    rsu.update_info()

            can_predict_current_tick = False
            if self.coperception_model_manager is not None and tick_number > 0:
                memory_structure = self._require_coperception_data_processor().build_live_memory(self.single_cav_list, self.rsu_list, tick_number)
                if memory_structure is not None:
                    self.coperception_model_manager.update_dataset(memory_structure)
                    opencood_dataset = self.coperception_model_manager.opencood_dataset
                    if opencood_dataset is None:
                        self._abort_simulation("Coperception dataset is missing; prediction pipeline cannot continue.")
                    opencood_dataset.extract_data(idx=0)
                    can_predict_current_tick = True

            opencda_message = payload_handler.make_opencda_message()
            logger.info(f"{round(opencda_message.ByteSize() / (1 << 20), 3)} MB of payload about to be sent")
            communication_manager.send_message(opencda_message)

            self.scenario_manager.sumo_tick()

            artery_message = communication_manager.receive_message()
            logger.info(f"{round(artery_message.ByteSize() / (1 << 20), 3)} MB were received")
            payload_handler.make_artery_payload(artery_message)

            if self.coperception_model_manager is not None and tick_number > 0 and can_predict_current_tick:
                self.coperception_model_manager.make_prediction(tick_number)

            payload_handler.clear_messages()

            if self.platoon_list is not None:
                for platoon in self.platoon_list:
                    platoon.run_step()

            new_messages: list[TransportMessage[Any]] = []
            if self.single_cav_list is not None:
                for single_cav in self.single_cav_list:
                    cav_messages, _ = single_cav.run_step(messages=self.messages)
                    new_messages.extend(cav_messages)

            if self.rsu_list is not None:
                for rsu in self.rsu_list:
                    rsu_messages, _ = rsu.run_step(messages=self.messages)
                    new_messages.extend(rsu_messages)

            self.simulation_snapshot = self._build_simulation_snapshot(tick_number)
            self.attack_results = self.attack_manager.evaluate(
                self.attacks,
                self.simulation_snapshot,
                service_resolver=self._require_cav_world().resolve_behavior_service,
            )
            for result in self.attack_results:
                logger.info("attack=%s status=%s reason=%s", result.attack_name, result.status.value, result.reason)

    def finalize(self, opt: argparse.Namespace) -> None:
        if opt.record:
            self.scenario_manager.client.stop_recorder()
            logger.info("finalizing: stopping recorder")

        if self.eval_manager is not None:
            self.eval_manager.evaluate(coperception_model_manager=self.coperception_model_manager)
            logger.info("finalizing: evaluating results")

        if self.single_cav_list is not None:
            logger.info(f"finalizing: destroying {len(self.single_cav_list)} single cavs")
            for v in self.single_cav_list:
                v.destroy()

        if self.rsu_list is not None:
            logger.info(f"finalizing: destroying {len(self.rsu_list)} RSUs")
            for r in self.rsu_list:
                r.destroy()

        if self.scenario_manager is not None:
            self.scenario_manager.close()
            logger.info("finalizing: evaluating results")

        if self.platoon_list is not None:
            logger.info(f"finalizing: destroying {len(self.platoon_list)} platoons")
            for platoon in self.platoon_list:
                platoon.destroy()

        if self.bg_veh_list is not None:
            logger.info(f"finalizing: destroying {len(self.bg_veh_list)} background cars")
            for v in self.bg_veh_list:
                v.destroy()

        if self.communication_manager:
            self.communication_manager.destroy()

        # TODO: Add general function to destroy actors


def run_scenario(opt: argparse.Namespace, scenario_params: DictConfig) -> None:
    raised_error: Exception | None = None
    scenario: Scenario | None = None
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
