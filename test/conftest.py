"""Pytest fixtures for unit tests.

Uses existing test/mocked_carla.py and adds:
- sys.modules["carla"] registration
- VehicleManager/RSUManager static state reset
- OpenCDA internal module stubs required to import tested modules
"""

from __future__ import annotations

import sys
import types
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from test import mocked_carla


def _install_stub(name: str, module: ModuleType) -> None:
    sys.modules[name] = module


def _make_placeholder_module(mod_name: str, **attrs) -> ModuleType:
    m = types.ModuleType(mod_name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass


# Install carla stub using existing mocked_carla classes
carla_stub = types.ModuleType("carla")
carla_stub.Location = mocked_carla.Location
carla_stub.Rotation = mocked_carla.Rotation
carla_stub.Transform = mocked_carla.Transform
carla_stub.Vector3D = mocked_carla.Vector3D
carla_stub.Vehicle = mocked_carla.Vehicle
carla_stub.BoundingBox = mocked_carla.BoundingBox

# Minimal placeholders for typing / attribute access
carla_stub.World = object
carla_stub.Map = object

# Callable (carla.Client(host, port)) – tests patch it anyway
carla_stub.Client = Mock


class WeatherParameters:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


carla_stub.WeatherParameters = WeatherParameters

# carla.command stub (needed by CoScenarioManager.spawn_actor)
command_stub = types.ModuleType("carla.command")


class SpawnActor:
    def __init__(self, blueprint, transform):
        self.blueprint = blueprint
        self.transform = transform

    def then(self, cmd):
        return self


class SetSimulatePhysics:
    def __init__(self, actor, enabled: bool):
        self.actor = actor
        self.enabled = enabled


command_stub.SpawnActor = SpawnActor
command_stub.SetSimulatePhysics = SetSimulatePhysics
command_stub.FutureActor = object()
carla_stub.command = command_stub

_install_stub("carla", carla_stub)
_install_stub("carla.command", command_stub)


# Optional dependency: omegaconf stub (only if not installed)
try:
    import omegaconf  # noqa: F401
except ImportError:  # pragma: no cover
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_listconfig_stub = types.ModuleType("omegaconf.listconfig")

    class OmegaConf:
        @staticmethod
        def create(obj):
            return obj

        @staticmethod
        def merge(*configs):
            result = {}
            for cfg in configs:
                if cfg and isinstance(cfg, dict):
                    result.update(cfg)
            return result

    omegaconf_stub.OmegaConf = OmegaConf
    omegaconf_listconfig_stub.ListConfig = list

    _install_stub("omegaconf", omegaconf_stub)
    _install_stub("omegaconf.listconfig", omegaconf_listconfig_stub)


# Stub OpenCDA internal modules used by vehicle_manager/rsu_manager/sim_api/cosim_api
_install_stub(
    "opencda.core.actuation.control_manager",
    _make_placeholder_module("opencda.core.actuation.control_manager", ControlManager=_Placeholder),
)
_install_stub(
    "opencda.core.application.platooning.platoon_behavior_agent",
    _make_placeholder_module("opencda.core.application.platooning.platoon_behavior_agent", PlatooningBehaviorAgent=_Placeholder),
)
_install_stub(
    "opencda.core.application.platooning.platooning_manager",
    _make_placeholder_module("opencda.core.application.platooning.platooning_manager", PlatooningManager=_Placeholder),
)
_install_stub(
    "opencda.core.common.v2x_manager",
    _make_placeholder_module("opencda.core.common.v2x_manager", V2XManager=_Placeholder),
)
_install_stub(
    "opencda.core.sensing.localization.localization_manager",
    _make_placeholder_module("opencda.core.sensing.localization.localization_manager", LocalizationManager=_Placeholder),
)
_install_stub(
    "opencda.core.sensing.localization.rsu_localization_manager",
    _make_placeholder_module("opencda.core.sensing.localization.rsu_localization_manager", LocalizationManager=_Placeholder),
)
_install_stub(
    "opencda.core.sensing.perception.perception_manager",
    _make_placeholder_module("opencda.core.sensing.perception.perception_manager", PerceptionManager=_Placeholder),
)
_install_stub(
    "opencda.core.safety.safety_manager",
    _make_placeholder_module("opencda.core.safety.safety_manager", SafetyManager=_Placeholder),
)
_install_stub(
    "opencda.core.plan.behavior_agent",
    _make_placeholder_module("opencda.core.plan.behavior_agent", BehaviorAgent=_Placeholder),
)
_install_stub(
    "opencda.core.map.map_manager",
    _make_placeholder_module("opencda.core.map.map_manager", MapManager=_Placeholder),
)
_install_stub(
    "opencda.core.common.data_dumper",
    _make_placeholder_module("opencda.core.common.data_dumper", DataDumper=_Placeholder),
)

_install_stub(
    "opencda.scenario_testing.utils.customized_map_api",
    _make_placeholder_module(
        "opencda.scenario_testing.utils.customized_map_api",
        load_customized_world=Mock(return_value=Mock()),
        bcolors=SimpleNamespace(FAIL="", ENDC=""),
    ),
)

_install_stub(
    "opencda.co_simulation.sumo_integration.constants",
    _make_placeholder_module(
        "opencda.co_simulation.sumo_integration.constants",
        SPAWN_OFFSET_Z=0.5,
        INVALID_ACTOR_ID=-1,
    ),
)
_install_stub(
    "opencda.co_simulation.sumo_integration.bridge_helper",
    _make_placeholder_module("opencda.co_simulation.sumo_integration.bridge_helper", BridgeHelper=Mock()),
)
_install_stub(
    "opencda.co_simulation.sumo_integration.sumo_simulation",
    _make_placeholder_module("opencda.co_simulation.sumo_integration.sumo_simulation", SumoSimulation=_Placeholder),
)


# Static state reset fixtures
@pytest.fixture(autouse=True)
def reset_vehicle_manager_state():
    from opencda.core.common.vehicle_manager import VehicleManager

    VehicleManager.current_cav_id = 1
    VehicleManager.current_platoon_id = 1
    VehicleManager.current_unknown_id = 1
    VehicleManager.used_ids = set()
    yield
    VehicleManager.current_cav_id = 1
    VehicleManager.current_platoon_id = 1
    VehicleManager.current_unknown_id = 1
    VehicleManager.used_ids = set()


@pytest.fixture(autouse=True)
def reset_rsu_manager_state():
    from opencda.core.common.rsu_manager import RSUManager

    RSUManager.current_id = 1
    RSUManager.used_ids = set()
    yield
    RSUManager.current_id = 1
    RSUManager.used_ids = set()


# Common fixtures
@pytest.fixture
def mock_cav_world():
    cav_world = Mock()
    cav_world.update_vehicle_manager = Mock()
    cav_world.update_rsu_manager = Mock()
    cav_world.update_sumo_vehicles = Mock()
    return cav_world


@pytest.fixture
def minimal_vehicle_config():
    return {
        "sensing": {"localization": {}, "perception": {}},
        "map_manager": {},
        "behavior": {},
        "controller": {},
        "v2x": {},
        "safety_manager": {},
        "platoon": {},
    }


@pytest.fixture
def minimal_rsu_config():
    return {
        "sensing": {"localization": {}, "perception": {}},
        "spawn_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
