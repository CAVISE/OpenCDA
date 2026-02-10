import sys
import types
from typing import Dict
from unittest.mock import MagicMock, Mock
import pytest

# Global storage for original modules to restore later
_ORIGINAL_MODULES = {}
_MOCKED_MODULE_NAMES = [
    "torch",
    "torch.cuda",
    "torch.utils",
    "torch.utils.data",
    "open3d",
    "open3d.visualization",
    "open3d.geometry",
    "opencood",
    "opencood.hypes_yaml",
    "opencood.hypes_yaml.yaml_utils",
    "opencood.tools",
    "opencood.tools.train_utils",
    "opencood.tools.inference_utils",
    "opencood.data_utils",
    "opencood.data_utils.datasets",
    "opencood.visualization",
    "opencood.visualization.simple_vis",
    "opencood.visualization.vis_utils",
    "opencood.utils",
    "opencood.utils.eval_utils",
    "tqdm",
]


def _install_mocks() -> None:
    """
    Installs mock modules into sys.modules.
    This is called via pytest_configure to ensure mocks exist before test collection imports.
    """
    # 1. Save original modules
    for mod_name in _MOCKED_MODULE_NAMES:
        if mod_name in sys.modules:
            _ORIGINAL_MODULES[mod_name] = sys.modules[mod_name]

    # 2. Mock torch
    torch = types.ModuleType("torch")
    torch.cuda = types.ModuleType("torch.cuda")
    torch.cuda.is_available = Mock(return_value=False)
    torch.device = Mock(side_effect=lambda x: f"device({x})")

    no_grad_mock = MagicMock()
    no_grad_mock.__enter__ = Mock()
    no_grad_mock.__exit__ = Mock()
    torch.no_grad = Mock(return_value=no_grad_mock)

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils

    class MockDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    torch_utils_data.DataLoader = MockDataLoader

    # 3. Mock open3d
    o3d = types.ModuleType("open3d")
    o3d.visualization = types.ModuleType("open3d.visualization")
    o3d.geometry = types.ModuleType("open3d.geometry")

    # Return distinct mock instances per call
    o3d.visualization.Visualizer = Mock(side_effect=lambda: MagicMock(name="VisualizerInstance"))
    o3d.geometry.PointCloud = Mock(side_effect=lambda: MagicMock(name="PointCloud"))
    o3d.geometry.LineSet = Mock(side_effect=lambda: MagicMock(name="LineSet"))

    # 4. Mock opencood and submodules
    opencood = types.ModuleType("opencood")

    # Submodules
    hypes_yaml = types.ModuleType("opencood.hypes_yaml")
    yaml_utils = types.ModuleType("opencood.hypes_yaml.yaml_utils")
    hypes_yaml.yaml_utils = yaml_utils
    opencood.hypes_yaml = hypes_yaml

    tools = types.ModuleType("opencood.tools")
    train_utils = types.ModuleType("opencood.tools.train_utils")
    inference_utils = types.ModuleType("opencood.tools.inference_utils")
    tools.train_utils = train_utils
    tools.inference_utils = inference_utils
    opencood.tools = tools

    data_utils = types.ModuleType("opencood.data_utils")
    datasets = types.ModuleType("opencood.data_utils.datasets")
    data_utils.datasets = datasets
    opencood.data_utils = data_utils

    visualization = types.ModuleType("opencood.visualization")
    simple_vis = types.ModuleType("opencood.visualization.simple_vis")
    vis_utils = types.ModuleType("opencood.visualization.vis_utils")
    visualization.simple_vis = simple_vis
    visualization.vis_utils = vis_utils
    opencood.visualization = visualization

    utils = types.ModuleType("opencood.utils")
    eval_utils = types.ModuleType("opencood.utils.eval_utils")
    utils.eval_utils = eval_utils
    opencood.utils = utils

    # --- Populate specific functions required by tests/production ---
    yaml_utils.load_yaml = Mock(return_value={})
    train_utils.create_model = Mock(return_value=MagicMock())
    train_utils.load_saved_model = Mock(return_value=(None, MagicMock()))
    train_utils.to_device = Mock(side_effect=lambda x, y: x)

    # Inference utils
    inference_utils.inference_late_fusion = Mock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    inference_utils.inference_early_fusion = Mock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    inference_utils.inference_intermediate_fusion = Mock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    inference_utils.save_prediction_gt = Mock()

    # Datasets
    datasets.build_dataset = Mock(return_value=MagicMock())

    # Vis
    simple_vis.visualize = Mock()
    vis_utils.visualize_inference_sample_dataloader = Mock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    vis_utils.linset_assign_list = Mock()

    # Eval
    eval_utils.caluclate_tp_fp = Mock()
    eval_utils.eval_final_results = Mock()

    # 5. Mock tqdm
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda x, **kwargs: x

    # Apply patches
    new_modules = {
        "torch": torch,
        "torch.cuda": torch.cuda,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "open3d": o3d,
        "open3d.visualization": o3d.visualization,
        "open3d.geometry": o3d.geometry,
        "opencood": opencood,
        "opencood.hypes_yaml": hypes_yaml,
        "opencood.hypes_yaml.yaml_utils": yaml_utils,
        "opencood.tools": tools,
        "opencood.tools.train_utils": train_utils,
        "opencood.tools.inference_utils": inference_utils,
        "opencood.data_utils": data_utils,
        "opencood.data_utils.datasets": datasets,
        "opencood.visualization": visualization,
        "opencood.visualization.simple_vis": simple_vis,
        "opencood.visualization.vis_utils": vis_utils,
        "opencood.utils": utils,
        "opencood.utils.eval_utils": eval_utils,
        "tqdm": tqdm_module,
    }
    sys.modules.update(new_modules)


def _uninstall_mocks() -> None:
    """
    Restores original modules or removes mocks.
    """
    for mod_name in _MOCKED_MODULE_NAMES:
        if mod_name in _ORIGINAL_MODULES:
            sys.modules[mod_name] = _ORIGINAL_MODULES[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


# Pytest hooks for setup/teardown at the start/end of the process
def pytest_configure(config) -> None:
    _install_mocks()


def pytest_unconfigure(config) -> None:
    _uninstall_mocks()


@pytest.fixture
def fake_heavy_deps() -> Dict[str, types.ModuleType]:
    """
    Returns the currently mocked modules from sys.modules for use in tests.
    """
    return {"torch": sys.modules["torch"], "opencood": sys.modules["opencood"], "open3d": sys.modules["open3d"]}
