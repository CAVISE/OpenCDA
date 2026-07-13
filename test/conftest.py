"""Pytest fixtures for unit tests.

Uses existing test/mocked_carla.py and adds:
- sys.modules["carla"] registration
- AgentManager static state reset
- OpenCDA internal module stubs required to import tested modules
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import importlib.util

import pytest

from test import mocked_carla


def _install_stub(name: str, module: ModuleType) -> None:
    sys.modules[name] = module


def _install_stub_if_missing(name: str, module: ModuleType) -> None:
    """Install stub module only if it's not already present (avoids duplication across conftest files)."""
    if name not in sys.modules:
        sys.modules[name] = module


behavior_services_stub = types.ModuleType("opencda.core.application.behavior.services")
behavior_services_stub.__all__ = []
behavior_services_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "opencda/core/application/behavior/services")]
_install_stub_if_missing("opencda.core.application.behavior.services", behavior_services_stub)


# Heavy external deps stubs for tests under test/ (torch/open3d/opencood may be absent in CI)
torch_stub = types.ModuleType("torch")


class _TorchTensor:
    pass


torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
torch_stub.hub = SimpleNamespace(load=Mock())
torch_stub.device = lambda *args, **kwargs: "cpu"
torch_stub.Tensor = _TorchTensor
torch_stub.nn = types.ModuleType("torch.nn")
torch_stub.nn.functional = types.ModuleType("torch.nn.functional")
torch_stub.nn.functional.affine_grid = lambda *args, **kwargs: Mock(name="affine_grid")()
torch_stub.nn.functional.grid_sample = lambda input_tensor, *args, **kwargs: input_tensor
torch_stub.optim = types.ModuleType("torch.optim")
torch_stub.optim.Adam = Mock
torch_stub.manual_seed = Mock()
torch_stub.from_numpy = lambda array: Mock(name="from_numpy_tensor")()
torch_stub.zeros = lambda *args, **kwargs: Mock(name="zeros_tensor")()
torch_stub.stack = lambda *args, **kwargs: Mock(name="stack_tensor")()
torch_stub.float32 = "float32"
_install_stub_if_missing("torch", torch_stub)
_install_stub_if_missing("torch.nn", torch_stub.nn)
_install_stub_if_missing("torch.nn.functional", torch_stub.nn.functional)
_install_stub_if_missing("torch.optim", torch_stub.optim)

_install_stub_if_missing("open3d", types.ModuleType("open3d"))
_install_stub_if_missing("opencood", types.ModuleType("opencood"))
opencood_utils_stub = types.ModuleType("opencood.utils")
opencood_box_utils_stub = types.ModuleType("opencood.utils.box_utils")
opencood_transformation_utils_stub = types.ModuleType("opencood.utils.transformation_utils")
opencood_transformation_utils_stub.x_to_world = lambda pose: [
    [1.0, 0.0, 0.0, float(pose[0])],
    [0.0, 1.0, 0.0, float(pose[1])],
    [0.0, 0.0, 1.0, float(pose[2])],
    [0.0, 0.0, 0.0, 1.0],
]
opencood_box_utils_stub.boxes_to_corners_3d = lambda boxes, order=None: boxes
opencood_box_utils_stub.boxes_to_corners2d = lambda boxes, order=None: boxes
opencood_utils_stub.box_utils = opencood_box_utils_stub
opencood_utils_stub.transformation_utils = opencood_transformation_utils_stub
_install_stub_if_missing("opencood.utils", opencood_utils_stub)
_install_stub_if_missing("opencood.utils.box_utils", opencood_box_utils_stub)
_install_stub_if_missing("opencood.utils.transformation_utils", opencood_transformation_utils_stub)


def _make_placeholder_module(mod_name: str, **attrs) -> ModuleType:
    m = types.ModuleType(mod_name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass


class _PlaceholderPerceptionRequirements:
    def __init__(
        self,
        enable_data_dump: bool = False,
        force_rgb_camera: bool = False,
        force_lidar: bool = False,
        force_semantic_lidar: bool = False,
        extend_inactive_detection_range: bool = False,
    ):
        self.enable_data_dump = enable_data_dump
        self.force_rgb_camera = force_rgb_camera
        self.force_lidar = force_lidar
        self.force_semantic_lidar = force_semantic_lidar
        self.extend_inactive_detection_range = extend_inactive_detection_range

    @classmethod
    def from_runtime_flags(cls, data_dump: bool = False, with_coperception: bool = False):
        return cls(
            enable_data_dump=data_dump,
            force_rgb_camera=data_dump,
            force_lidar=data_dump or with_coperception,
            force_semantic_lidar=data_dump or with_coperception,
            extend_inactive_detection_range=data_dump or with_coperception,
        )


# Install carla stub using existing mocked_carla classes
carla_stub = types.ModuleType("carla")
carla_stub.Location = mocked_carla.Location
carla_stub.Rotation = mocked_carla.Rotation
carla_stub.Transform = mocked_carla.Transform
carla_stub.Vector3D = mocked_carla.Vector3D
carla_stub.Vehicle = mocked_carla.Vehicle
carla_stub.BoundingBox = mocked_carla.BoundingBox

# Some OpenCDA modules use runtime-evaluated type annotations like `carla.Actor`
# and `carla.Color`. These attributes must exist on the stub to avoid import-time
# failures.
carla_stub.Actor = object


class AttachmentType:
    Rigid = object()


carla_stub.AttachmentType = AttachmentType


class Color:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


carla_stub.Color = Color


# Placeholder to satisfy type annotations and basic attribute access.
class TrafficLightState:
    Red = 0
    Yellow = 1
    Green = 2
    Off = 3
    Unknown = 4


carla_stub.TrafficLightState = TrafficLightState

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
if importlib.util.find_spec("omegaconf") is None:  # pragma: no cover
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


# Stub OpenCDA internal modules used by agent_manager/sim_api/cosim_api
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
    "opencda.core.sensing.perception.perception_manager",
    _make_placeholder_module(
        "opencda.core.sensing.perception.perception_manager",
        PerceptionManager=_Placeholder,
        PerceptionRequirements=_PlaceholderPerceptionRequirements,
    ),
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


# Stubbing opencda.core.application.behavior.services above blocks the package's
# auto-discovery (which would otherwise need carla at import time). The services
# used by tests must therefore be imported explicitly so their @BehaviorServiceRegistry.register
# decorators run.
import importlib as _importlib  # noqa: E402

_importlib.import_module("opencda.core.application.behavior.services.movement_controller")
_importlib.import_module("opencda.core.application.behavior.services.default_movement_request")


# Static state reset fixture
@pytest.fixture(autouse=True)
def reset_agent_manager_state():
    from opencda.core.common.agent_manager import AgentManager

    AgentManager.reset_id_registry()
    yield
    AgentManager.reset_id_registry()


# Common fixtures
@pytest.fixture
def mock_cav_world():
    cav_world = Mock()
    cav_world.update_agent_manager = Mock()
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
        "behavior_services": [
            {"type": "default_movement_request"},
            {"type": "movement_controller"},
        ],
    }


@pytest.fixture
def minimal_rsu_config():
    return {
        "sensing": {"localization": {}, "perception": {}},
        "spawn_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


@pytest.fixture(autouse=True)
def disable_cv2_gui(monkeypatch):
    """
    Ensure unit tests never open GUI windows in headless environments.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return

    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None, raising=True)
    monkeypatch.setattr(cv2, "waitKey", lambda *args, **kwargs: 1, raising=True)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *args, **kwargs: None, raising=True)
