from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NoReturn, cast

import carla
from omegaconf import DictConfig, OmegaConf

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.application.platooning.platooning_manager import PlatooningManager
from opencda.core.attack.adversary_framework import Attack, AttackManager, AttackSpec
from opencda.core.common.agent_manager import AgentManager
from opencda.core.common.cav_world import CavWorld
from opencda.core.common.coperception_data_processor import CoperceptionDataProcessor
from opencda.core.common.tick_profiler import TickProfiler
from opencda.core.sensing.perception.perception_manager import PerceptionRequirements
from opencda.metrics_tools.metric_collector import MetricCollector
from opencda.scenario_testing.types import NodeSnapshot, SimulationSnapshot
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import YamlDict, add_current_time, load_yaml, save_yaml

from opencda.core.application.behavior import BROADCAST_OWNER_ID, TransportMessage

if TYPE_CHECKING:
    from opencda.core.common.coperception_model_manager import CoperceptionModelManager
    from opencda.core.common.communication.communication_manager import CommunicationManager
    from opencda.core.common.communication.payload_handler import PayloadHandler

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.scenario")


@dataclass
class Scenario:
    eval_manager: EvaluationManager
    scenario_manager: sim_api.ScenarioManager | sim_api.CoScenarioManager
    single_cav_list: list[AgentManager]
    rsu_list: list[AgentManager]
    spectator: carla.Actor
    cav_world: CavWorld
    platoon_list: list[PlatooningManager]
    bg_veh_list: list[carla.Actor]
    scenario_name: str
    tick_profiler: TickProfiler

    def _abort_simulation(self, message: str) -> NoReturn:
        logger.error(message)
        raise RuntimeError(message)

    def _require_coperception_data_processor(self) -> CoperceptionDataProcessor:
        if self.coperception_data_processor is None:
            self._abort_simulation("Coperception data processor is required, but it was not initialized.")
        return self.coperception_data_processor

    def __init__(self, opt: argparse.Namespace, scenario_params: DictConfig) -> None:
        self.node_ids: dict[str, dict[int, str]] = {"cav": {}, "rsu": {}, "platoon": {}}
        self.scenario_name = opt.test_scenario
        self.tick_profiler = TickProfiler(enabled=getattr(opt, "tick_timing", False))
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
        self.cav_world = self.scenario_manager.cav_world

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

        self.single_cav_list, cav_node_ids = self.scenario_manager.create_vehicle_agents(
            application=["single"],
            map_helper=map_api.spawn_helper_2lanefree,
            perception_requirements=perception_requirements,
        )
        self.node_ids["cav"] = cast(dict[int, str], cav_node_ids)
        logger.info(f"created single cavs of size {len(self.single_cav_list)}")

        _, self.bg_veh_list = self.scenario_manager.create_traffic_carla()
        logger.info(f"created background traffic of size {len(self.bg_veh_list)}")

        self.rsu_list, rsu_node_ids = self.scenario_manager.create_rsu_agents(
            perception_requirements=perception_requirements,
        )
        self.node_ids["rsu"] = cast(dict[int, str], rsu_node_ids)
        logger.info(f"created RSU list of size {len(self.rsu_list)}")

        if opt.with_coperception and opt.model_dir:
            if not os.path.isdir(opt.model_dir):
                self._abort_simulation(f'Model directory "{opt.model_dir}" does not exist; cannot initialize cooperative perception manager.')

            CoperceptionManagerClass: type[CoperceptionModelManager]
            if opt.with_advcp:
                from opencda.core.attack.advcp.adv_coperception_model_manager import AdvCoperceptionModelManager as CoperceptionManagerClass
            else:
                from opencda.core.common.coperception_model_manager import CoperceptionModelManager as CoperceptionManagerClass

            coperception_config = OmegaConf.to_container(
                scenario_params.get("coperception", {}),
                resolve=True,
            )

            self.coperception_model_manager = CoperceptionManagerClass(
                opt=opt,
                current_time=current_time,
                payload_handler=self.payload_handler,
                coperception_config=coperception_config,
            )
            valid_agent_ids = [vehicle_manager.id for vehicle_manager in self.single_cav_list]
            valid_agent_ids.extend(rsu_manager.id for rsu_manager in self.rsu_list)
            if hasattr(self.coperception_model_manager, "validate_advcp_agents"):
                self.coperception_model_manager.validate_advcp_agents(valid_agent_ids)

            """
            TODO: Create decorators to write such stuff

            @cavise.SimObject
            class SomeCoolManager
            that would at least take the logging part - writing "creating SomeCoolManager manager", destroying SomeCoolManager manager"

            Also ideally it would also somehow verify manager configs, for example.
            """
            sensor_sync_config = coperception_config.get("sensor_sync", {}) if isinstance(coperception_config, Mapping) else {}
            sensor_sync_timeout = float(sensor_sync_config.get("timeout_seconds", 1.0)) if isinstance(sensor_sync_config, Mapping) else 1.0
            self.coperception_data_processor = CoperceptionDataProcessor(sensor_sync_timeout_seconds=sensor_sync_timeout)
            logger.info("created cooperception manager")

        self.scenario_manager.create_custom_actor_manager(application=["single"], map_helper=map_api.spawn_helper_2lanefree, data_dump=opt.record)
        logger.info("created single custom actors")

        self.scenario_manager.tick()
        logger.info("completed initial spawn synchronization tick")

        self.eval_manager = EvaluationManager(self.cav_world, script_name=self.scenario_name, current_time=scenario_config["current_time"])

        self.spectator = self.scenario_manager.world.get_spectator()

        self.messages: list[TransportMessage[Any]] = []
        self.simulation_snapshot = SimulationSnapshot(tick=-1)
        self.attack_manager = AttackManager()
        self.scenario_metrics_collector = MetricCollector(
            module="scenario",
            entity_id=self.scenario_name,
            metric_configs={
                "collision_count": {"warmup_steps": 0},
                "near_miss_count": {"warmup_steps": 0},
                "identity_conflict_count": {"warmup_steps": 0},
            },
        )
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

    def _build_simulation_snapshot(self, tick: int) -> SimulationSnapshot:
        vehicle_nodes = tuple(
            NodeSnapshot(
                node_id=vehicle_manager.id,
                node_type="vehicle",
                service_states=dict(vehicle_manager.behavior_service_states),
            )
            for vehicle_manager in chain(
                self.single_cav_list,
                *(platoon.agent_manager_list for platoon in self.platoon_list),
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

    def _collect_safety_status(self, vehicle_manager: AgentManager) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for sensor in vehicle_manager.agent.safety_manager.sensors:
            return_status = getattr(sensor, "return_status", None)
            if not callable(return_status):
                continue

            sensor_status = return_status()
            if isinstance(sensor_status, dict):
                status.update({str(key): bool(value) for key, value in sensor_status.items()})
        return status

    def _build_scenario_metric_context(self, identity_claims: tuple[Mapping[str, str], ...]) -> dict[str, Any]:
        vehicles: list[dict[str, Any]] = []

        for vehicle_manager in chain(
            self.single_cav_list,
            *(platoon.agent_manager_list for platoon in self.platoon_list),
        ):
            ego_pos = vehicle_manager.agent.localizer.get_state().transform

            safety_status = self._collect_safety_status(vehicle_manager)
            vehicles.append(
                {
                    "node_id": vehicle_manager.id,
                    "x": ego_pos.location.x,
                    "y": ego_pos.location.y,
                    "z": ego_pos.location.z,
                    "collided": bool(safety_status.get("collision", False)),
                }
            )

        return {
            "vehicles": tuple(vehicles),
            "identity_claims": identity_claims,
        }

    @staticmethod
    def _collect_identity_claims(
        producer_node_id: str,
        messages: list[TransportMessage[Any]],
    ) -> tuple[dict[str, str], ...]:
        claims: list[dict[str, str]] = []

        for message in messages:
            claimed_node_id = getattr(message, "src_owner_id", "")
            if not isinstance(claimed_node_id, str) or not claimed_node_id or claimed_node_id == BROADCAST_OWNER_ID:
                continue

            claims.append(
                {
                    "producer_node_id": producer_node_id,
                    "claimed_node_id": claimed_node_id,
                }
            )

        return tuple(claims)

    def run(self, opt: argparse.Namespace) -> None:
        if self.communication_manager is None:
            self.default_loop(opt)
        else:
            self.capi_loop(opt)

    def default_loop(self, opt: argparse.Namespace) -> None:
        tick_number = -1
        profiler = self.tick_profiler
        agent_profiler = profiler if profiler.enabled else None
        while True:
            tick_number += 1

            if opt.ticks and tick_number > opt.ticks:
                break
            logger.debug(f"running: simulation tick: {tick_number}")
            profiler.start_tick(tick_number)
            with profiler.measure("sumo_tick"):
                self.scenario_manager.sumo_tick()
            with profiler.measure("carla_tick"):
                carla_frame = self.scenario_manager.tick()
            with profiler.measure("world_frame"):
                world_frame = self.scenario_manager.capture_world_frame(carla_frame)

            with profiler.measure("spectator"):
                if not opt.free_spectator and any(array is not None for array in [self.single_cav_list, self.platoon_list]):
                    if len(self.single_cav_list) > 0:
                        transform = world_frame.actor_state(self.single_cav_list[0].agent.vehicle.id).transform
                        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
                    else:
                        platoon_vehicle = self.platoon_list[0].agent_manager_list[0].agent.vehicle
                        transform = world_frame.actor_state(platoon_vehicle.id).transform
                        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            with profiler.measure("platoons"):
                if self.platoon_list is not None:
                    logger.debug("updating platoons")
                    for platoon in self.platoon_list:
                        platoon.update_information()

            new_messages: list[TransportMessage[Any]] = []
            identity_claims: list[Mapping[str, str]] = []
            if self.single_cav_list is not None:
                logger.debug("updating single cavs")

                with profiler.measure("cav_update"):
                    for single_cav in self.single_cav_list:
                        single_cav.agent.update(world_frame, agent_profiler)

            if self.rsu_list is not None:
                logger.debug("updating RSUs")
                with profiler.measure("rsu_update"):
                    for rsu in self.rsu_list:
                        rsu.agent.update(world_frame, agent_profiler)

            with profiler.measure("coperception"):
                if self.coperception_model_manager is not None and tick_number > 0:
                    logger.info(f"Processing {tick_number} tick")
                    memory_structure = self._require_coperception_data_processor().build_live_memory(
                        self.single_cav_list,
                        self.rsu_list,
                        tick_number,
                        sensor_frame=carla_frame,
                    )
                    if memory_structure is None:
                        logger.warning(f"Live cooperative perception data for tick {tick_number} is not available.")
                    else:
                        self.coperception_model_manager.update_dataset(memory_structure)
                        self.coperception_model_manager.make_prediction(tick_number)
                        logger.info(f"Successfully processed {tick_number} tick")

            with profiler.measure("platoons"):
                if self.platoon_list is not None:
                    for platoon in self.platoon_list:
                        platoon.run_step()

            if self.single_cav_list is not None:
                for single_cav in self.single_cav_list:
                    if profiler.enabled:
                        with profiler.measure("behavior_services"):
                            cav_messages, _ = single_cav.update_behavior_services(self.messages)
                        with profiler.measure("finish_step"):
                            single_cav.agent.finish_step()
                    else:
                        cav_messages, _ = single_cav.update_behavior_services(self.messages)
                        single_cav.agent.finish_step()
                    new_messages.extend(cav_messages)
                    identity_claims.extend(self._collect_identity_claims(single_cav.id, cav_messages))
            else:
                identity_claims = []

            if self.rsu_list is not None:
                for rsu in self.rsu_list:
                    if profiler.enabled:
                        with profiler.measure("behavior_services"):
                            rsu_messages, _ = rsu.update_behavior_services(self.messages)
                        with profiler.measure("finish_step"):
                            rsu.agent.finish_step()
                    else:
                        rsu_messages, _ = rsu.update_behavior_services(self.messages)
                        rsu.agent.finish_step()
                    new_messages.extend(rsu_messages)
                    identity_claims.extend(self._collect_identity_claims(rsu.id, rsu_messages))

            with profiler.measure("post_update"):
                self.messages = new_messages
                self.simulation_snapshot = self._build_simulation_snapshot(tick_number)
                self.scenario_metrics_collector.update(self._build_scenario_metric_context(tuple(identity_claims)))

                self.attack_manager.evaluate(
                    self.attacks,
                    self.simulation_snapshot,
                    service_resolver=self.cav_world.resolve_behavior_services,
                )
            profiler.finish_tick()

    def capi_loop(self, opt: argparse.Namespace) -> None:
        if self.communication_manager is None:
            self._abort_simulation("CommunicationManager is required for CAPI flow, but it was not initialized.")
        if self.payload_handler is None:
            self._abort_simulation("PayloadHandler is required for CAPI flow, but it was not initialized.")
        communication_manager = self.communication_manager
        payload_handler = self.payload_handler
        tick_number = -1
        profiler = self.tick_profiler
        agent_profiler = profiler if profiler.enabled else None
        while True:
            tick_number += 1

            if opt.ticks and tick_number > opt.ticks:
                break
            logger.debug(f"running: simulation tick: {tick_number}")
            profiler.start_tick(tick_number)
            with profiler.measure("carla_tick"):
                carla_frame = self.scenario_manager.tick()
            with profiler.measure("world_frame"):
                world_frame = self.scenario_manager.capture_world_frame(carla_frame)

            with profiler.measure("spectator"):
                if not opt.free_spectator and any(array is not None for array in [self.single_cav_list, self.platoon_list]):
                    if len(self.single_cav_list) > 0:
                        transform = world_frame.actor_state(self.single_cav_list[0].agent.vehicle.id).transform
                        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
                    else:
                        platoon_vehicle = self.platoon_list[0].agent_manager_list[0].agent.vehicle
                        transform = world_frame.actor_state(platoon_vehicle.id).transform
                        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            # TODO: Add aim service support

            with profiler.measure("platoons"):
                if self.platoon_list is not None:
                    logger.debug("updating platoons")
                    for platoon in self.platoon_list:
                        platoon.update_information()

            if self.single_cav_list is not None:
                logger.debug("updating single cavs")
                with profiler.measure("cav_update"):
                    for single_cav in self.single_cav_list:
                        single_cav.agent.update(world_frame, agent_profiler)

            if self.rsu_list is not None:
                logger.debug("updating RSUs")
                with profiler.measure("rsu_update"):
                    for rsu in self.rsu_list:
                        rsu.agent.update(world_frame, agent_profiler)

            can_predict_current_tick = False
            with profiler.measure("coperception"):
                if self.coperception_model_manager is not None and tick_number > 0:
                    memory_structure = self._require_coperception_data_processor().build_live_memory(
                        self.single_cav_list,
                        self.rsu_list,
                        tick_number,
                        sensor_frame=carla_frame,
                    )
                    if memory_structure is not None:
                        self.coperception_model_manager.update_dataset(memory_structure)
                        opencood_dataset = self.coperception_model_manager.opencood_dataset
                        if opencood_dataset is None:
                            self._abort_simulation("Coperception dataset is missing; prediction pipeline cannot continue.")
                        opencood_dataset.extract_data(idx=0)
                        can_predict_current_tick = True

            with profiler.measure("capi_exchange"):
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

            with profiler.measure("platoons"):
                if self.platoon_list is not None:
                    for platoon in self.platoon_list:
                        platoon.run_step()

            new_messages: list[TransportMessage[Any]] = []
            identity_claims: list[Mapping[str, str]] = []
            if self.single_cav_list is not None:
                for single_cav in self.single_cav_list:
                    if profiler.enabled:
                        with profiler.measure("behavior_services"):
                            cav_messages, _ = single_cav.update_behavior_services(self.messages)
                        with profiler.measure("finish_step"):
                            single_cav.agent.finish_step()
                    else:
                        cav_messages, _ = single_cav.update_behavior_services(self.messages)
                        single_cav.agent.finish_step()
                    new_messages.extend(cav_messages)
                    identity_claims.extend(self._collect_identity_claims(single_cav.id, cav_messages))

            if self.rsu_list is not None:
                for rsu in self.rsu_list:
                    if profiler.enabled:
                        with profiler.measure("behavior_services"):
                            rsu_messages, _ = rsu.update_behavior_services(self.messages)
                        with profiler.measure("finish_step"):
                            rsu.agent.finish_step()
                    else:
                        rsu_messages, _ = rsu.update_behavior_services(self.messages)
                        rsu.agent.finish_step()
                    new_messages.extend(rsu_messages)
                    identity_claims.extend(self._collect_identity_claims(rsu.id, rsu_messages))

            with profiler.measure("post_update"):
                self.messages = new_messages
                self.simulation_snapshot = self._build_simulation_snapshot(tick_number)
                self.scenario_metrics_collector.update(self._build_scenario_metric_context(tuple(identity_claims)))
                self.attack_manager.evaluate(
                    self.attacks,
                    self.simulation_snapshot,
                    service_resolver=self.cav_world.resolve_behavior_services,
                )
            profiler.finish_tick()

    def finalize(self, opt: argparse.Namespace) -> None:
        try:
            self._stop_runtime_sensors()

            if opt.record:
                self.scenario_manager.client.stop_recorder()
                logger.info("finalizing: stopping recorder")

            if self.eval_manager is not None:
                self.eval_manager.evaluate(
                    coperception_model_manager=self.coperception_model_manager,
                    scenario_metrics_collector=self.scenario_metrics_collector,
                )
                logger.info("finalizing: evaluating results")
        finally:
            finalization_failed = sys.exc_info()[0] is not None
            try:
                self._destroy_resources()
            except Exception:
                if finalization_failed:
                    logger.exception("Resource cleanup also failed during scenario finalization.")
                else:
                    raise

    def _stop_runtime_sensors(self) -> None:
        vehicle_managers = chain(
            self.single_cav_list or (),
            *(platoon.agent_manager_list for platoon in (self.platoon_list or ())),
        )
        for manager in vehicle_managers:
            try:
                manager.agent.stop_runtime_sensors()
            except Exception:
                logger.exception("Failed to stop runtime sensors for CAV %s.", manager.id)

    def _destroy_resources(self) -> None:
        first_exception: Exception | None = None

        def destroy(resource: Any, description: str) -> None:
            nonlocal first_exception
            try:
                resource.destroy()
            except Exception as exc:
                logger.exception("Failed to destroy %s.", description)
                if first_exception is None:
                    first_exception = exc

        logger.info("finalizing: destroying %d single cavs", len(self.single_cav_list or ()))
        for manager in self.single_cav_list or ():
            destroy(manager, f"single CAV {manager.id}")

        logger.info("finalizing: destroying %d RSUs", len(self.rsu_list or ()))
        for manager in self.rsu_list or ():
            destroy(manager, f"RSU {manager.id}")

        if self.scenario_manager is not None:
            try:
                self.scenario_manager.close()
                logger.info("finalizing: closing scenario manager")
            except Exception as exc:
                logger.exception("Failed to close scenario manager.")
                if first_exception is None:
                    first_exception = exc

        logger.info("finalizing: destroying %d platoons", len(self.platoon_list or ()))
        for platoon in self.platoon_list or ():
            destroy(platoon, "platoon")

        logger.info("finalizing: destroying %d background cars", len(self.bg_veh_list or ()))
        for vehicle in self.bg_veh_list or ():
            destroy(vehicle, f"background vehicle {vehicle.id}")

        if self.communication_manager is not None:
            destroy(self.communication_manager, "communication manager")

        if first_exception is not None:
            raise first_exception


def run_scenario(opt: argparse.Namespace, scenario_params: DictConfig) -> None:
    raised_error: Exception | None = None
    finalization_error: Exception | None = None
    scenario: Scenario | None = None
    try:
        scenario = Scenario(opt, scenario_params)
        scenario.run(opt)
    except Exception as error:
        logger.exception("Simulation failed before finalization.")
        raised_error = error
    finally:
        logger.info("Wrapping things up... Please don't press Ctrl+C")
        if scenario:
            try:
                scenario.finalize(opt)
            except Exception as error:
                logger.exception("Scenario finalization failed.")
                finalization_error = error
        if raised_error is not None:
            raise raised_error
        if finalization_error is not None:
            raise finalization_error
