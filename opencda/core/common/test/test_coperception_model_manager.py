import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from importlib import import_module
import numpy as np

# The production code imports are now safe because pytest_configure in conftest.py
# installs the mocks before collection.
from opencda.core.common.coperception_data_processor import CoperceptionDataProcessor
from opencda.core.common.coperception_model_manager import (
    CoperceptionModelManager,
    CoperceptionVisualizer,
)


class DummyOpt:
    def __init__(self, **kwargs):
        self.model_dir = "test_model_dir"
        self.fusion_method = "late"
        self.show_video_vis = False
        self.save_npy = False
        self.save_vis = False
        self.test_scenario = "test_scenario"
        self.__dict__.update(kwargs)


class DummyDataset:
    def __init__(self):
        self.data = []

    def __len__(self):
        return 10

    def collate_batch_test(self, batch):
        return batch

    def visualize_result(self, *args, **kwargs):
        pass

    def update_database(self, memory_data=None):
        pass


class TestCoperceptionModelManager:
    @pytest.fixture
    def manager_deps(self, fake_heavy_deps):
        """
        Setup mocks specifically for Manager instantiation and method calls.
        Resets mocks before every test to ensure isolation.
        """
        opencood = fake_heavy_deps["opencood"]
        torch = fake_heavy_deps["torch"]
        open3d = fake_heavy_deps["open3d"]

        # Shortcuts to specific mocks (Modules & Objects)
        # Note: We cannot call reset_mock() on modules, only on Mocks.

        mocks_to_reset = [
            opencood.hypes_yaml.yaml_utils.load_yaml,
            opencood.tools.train_utils.create_model,
            opencood.tools.train_utils.load_saved_model,
            opencood.tools.train_utils.to_device,
            opencood.tools.inference_utils.inference_late_fusion,
            opencood.tools.inference_utils.inference_early_fusion,
            opencood.tools.inference_utils.inference_intermediate_fusion,
            opencood.tools.inference_utils.save_prediction_gt,
            opencood.data_utils.datasets.build_dataset,
            opencood.models.communication_adapters.build_communication_adapter,
            opencood.models.communication_adapters.build_communication_adapter.return_value,
            opencood.visualization.simple_vis.visualize,
            opencood.visualization.vis_utils.visualize_inference_sample_dataloader,
            opencood.visualization.vis_utils.linset_assign_list,
            opencood.utils.eval_utils.caluclate_tp_fp,
            opencood.utils.eval_utils.calculate_ap,
            opencood.utils.eval_utils.eval_final_results,
            open3d.visualization.Visualizer,
            torch.cuda.is_available,
            torch.device,
            torch.no_grad,
        ]

        # Reset actual mock objects
        for m in mocks_to_reset:
            m.reset_mock()

        # Remove side effects from previous tests
        opencood.utils.eval_utils.caluclate_tp_fp.side_effect = None

        # Setup default return values
        hypes = {
            "postprocess": {"core_method": "VoxelPostprocessor", "gt_range": [0, -40, -3, 70, 40, 1]},
            "fusion": {"core_method": "LateFusionDataset"},
        }
        opencood.hypes_yaml.yaml_utils.load_yaml.return_value = hypes

        model = MagicMock()
        opencood.data_utils.datasets.build_dataset.return_value = MagicMock()
        opencood.models.communication_adapters.build_communication_adapter.return_value = MagicMock()
        opencood.tools.train_utils.create_model.return_value = model
        opencood.tools.train_utils.load_saved_model.return_value = (None, model)

        # Re-bind imported module-level dependencies to this fixture's mocks.
        # This keeps tests deterministic even when other test trees import the
        # manager module before these conftest mocks are installed.
        coperception_model_manager_module = import_module(CoperceptionModelManager.__module__)
        coperception_model_manager_module.torch = torch
        coperception_model_manager_module.o3d = open3d
        coperception_model_manager_module.yaml_utils = opencood.hypes_yaml.yaml_utils
        coperception_model_manager_module.train_utils = opencood.tools.train_utils
        coperception_model_manager_module.inference_utils = opencood.tools.inference_utils
        coperception_model_manager_module.build_dataset = opencood.data_utils.datasets.build_dataset
        coperception_model_manager_module.build_communication_adapter = opencood.models.communication_adapters.build_communication_adapter
        coperception_model_manager_module.vis_utils = opencood.visualization.vis_utils

        # Return dict for easy access
        return {
            "yaml_utils": opencood.hypes_yaml.yaml_utils,
            "train_utils": opencood.tools.train_utils,
            "inference_utils": opencood.tools.inference_utils,
            "vis_utils": opencood.visualization.vis_utils,
            "simple_vis": opencood.visualization.simple_vis,
            "eval_utils": opencood.utils.eval_utils,
            "build_dataset": opencood.data_utils.datasets.build_dataset,
            "build_communication_adapter": (opencood.models.communication_adapters.build_communication_adapter),
            "Visualizer": open3d.visualization.Visualizer,
            "torch": torch,
            "model": model,
            "hypes": hypes,
        }

    def test_init_cpu(self, manager_deps):
        manager_deps["torch"].cuda.is_available.return_value = False
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")

        assert manager.device == "device(cpu)"
        manager_deps["model"].cuda.assert_not_called()
        manager_deps["train_utils"].load_saved_model.assert_called_with("test_model_dir", manager_deps["model"])

    def test_init_cuda(self, manager_deps):
        manager_deps["torch"].cuda.is_available.return_value = True
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")

        assert manager.device == "device(cuda)"
        manager_deps["model"].cuda.assert_called_once()

    def test_init_attaches_model_communication_adapter_to_dataset(self, manager_deps):
        dataset = DummyDataset()
        adapter = MagicMock()
        manager_deps["build_dataset"].return_value = dataset
        manager_deps["build_communication_adapter"].return_value = adapter

        manager = CoperceptionModelManager(DummyOpt(), "2023_01_01")

        manager_deps["build_communication_adapter"].assert_called_once_with(
            manager.model,
            manager.device,
            "LateFusionDataset",
        )
        assert manager.communication_adapter is adapter
        assert dataset.communication_adapter is adapter

    def test_prepare_transmission_payloads_delegates_to_adapter(self, manager_deps):
        manager = CoperceptionModelManager(DummyOpt(), "2023_01_01")

        manager.prepare_transmission_payloads(idx=7)

        manager.communication_adapter.prepare_transmission_payloads.assert_called_once_with(
            manager.opencood_dataset,
            7,
        )

    def test_prepare_transmission_payloads_requires_dataset(self, manager_deps):
        manager = CoperceptionModelManager(DummyOpt(), "2023_01_01")
        manager.opencood_dataset = None

        with pytest.raises(RuntimeError, match="dataset is missing"):
            manager.prepare_transmission_payloads(idx=0)

    def test_coperception_metrics_use_scenario_config(self, manager_deps):
        opt = DummyOpt()
        manager = CoperceptionModelManager(
            opt,
            "2023_01_01",
            coperception_config={
                "visualization": {
                    "background": [1, 2, 3],
                },
                "metrics": {
                    "metric_configs": {
                        "ap_at_iou": {"warmup_steps": 5, "global_sort_detections": True},
                        "attack_success_rate": {"warmup_steps": 7, "iou_threshold": 0.5},
                        "attacker_benign_visibility_ratio": {"warmup_steps": 2, "epsilon": 2.0},
                    }
                },
            },
        )

        assert manager.metrics_collector.active_metrics == [
            "ap_at_iou",
            "attack_success_rate",
            "attacker_benign_visibility_ratio",
        ]
        assert manager.metrics_collector.metrics["ap_at_iou"].warmup_steps == 5
        assert manager.metrics_collector.metrics["ap_at_iou"].global_sort_detections is True
        assert manager.metrics_collector.metrics["attack_success_rate"].warmup_steps == 7
        assert manager.metrics_collector.metrics["attack_success_rate"].iou_threshold == 0.5
        visibility_metric = manager.metrics_collector.metrics["attacker_benign_visibility_ratio"]
        assert visibility_metric.warmup_steps == 2
        assert visibility_metric.epsilon == 2.0
        assert manager.visualization_config["background"] == (1, 2, 3)

    def test_update_dataset(self, manager_deps):
        """
        Verify update_dataset calls the correct build_dataset and creates a DataLoader.
        We patch the symbol inside the module under test to ensure we capture the call.
        """
        dataset_mock = DummyDataset()
        dataset_mock.update_database = MagicMock()
        memory_data = {0: {"cav_1": {"000010": {"yaml": "a.yaml", "lidar": "a.pcd"}}}}

        # Patching where it is imported in the source code
        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock) as mock_build:
            opt = DummyOpt()
            manager = CoperceptionModelManager(opt, "2023_01_01")

            manager.update_dataset(data=memory_data)

            dataset_mock.update_database.assert_called_once_with(memory_data=memory_data)
            mock_build.assert_called_with(manager_deps["hypes"], visualize=True, train=False, communication_interface=None)
            assert manager.opencood_dataset == dataset_mock
            assert manager.data_loader is not None
            assert manager.data_loader.dataset == dataset_mock

    def test_update_dataset_logs_warning_for_empty_dataset(self):
        """Verify warning is emitted when update leads to an empty dataset."""
        dataset_mock = MagicMock()
        dataset_mock.collate_batch_test = MagicMock()
        dataset_mock.__len__.return_value = 0

        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock):
            manager = CoperceptionModelManager(DummyOpt(), "2023_01_01")

        with patch("opencda.core.common.coperception_model_manager.logger.warning") as mock_warning:
            manager.update_dataset(data={"in_memory": True})

        dataset_mock.update_database.assert_called_once_with(memory_data={"in_memory": True})
        mock_warning.assert_called_once_with("No samples found in dataset after update.")

    @pytest.mark.parametrize(
        ("core_method", "inference_attr"),
        [
            ("LateFusionDataset", "inference_late_fusion"),
            ("EarlyFusionDataset", "inference_early_fusion"),
            ("IntermediateFusionDataset", "inference_intermediate_fusion"),
            ("IntermediateFusionDatasetV2", "inference_intermediate_fusion"),
        ],
    )
    def test_make_prediction_fusion_methods(self, core_method, inference_attr, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = core_method
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = [{"ego": {"origin_lidar": ["lidar_data"]}}]
        manager.opencood_dataset = MagicMock()

        manager.make_prediction(0)

        getattr(manager_deps["inference_utils"], inference_attr).assert_called()

    def test_init_raises_for_unsupported_core_method(self, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = "BrokenFusionDataset"

        with pytest.raises(NotImplementedError, match="Unsupported cooperative perception fusion.core_method"):
            CoperceptionModelManager(DummyOpt(), "2023_01_01")

    def test_make_prediction_save_npy(self, manager_deps, tmp_path, monkeypatch):
        """Test saving NPY files using real filesystem operations in tmp_path."""
        monkeypatch.chdir(tmp_path)

        opt = DummyOpt(save_npy=True, test_scenario="scen1")
        manager = CoperceptionModelManager(opt, "2023_01_01")

        manager.data_loader = [{"ego": {"origin_lidar": ["lidar"]}}]
        manager.opencood_dataset = MagicMock()
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("p", "s", "g")

        manager.make_prediction(10)

        # Check directory creation
        expected_dir = tmp_path / "simulation_output/coperception/npy/scen1_2023_01_01/npy"
        assert expected_dir.exists()

        # Check call
        manager_deps["inference_utils"].save_prediction_gt.assert_called()
        args = manager_deps["inference_utils"].save_prediction_gt.call_args[0]
        assert args[3] == 10
        # args[4] is the path passed to save_prediction_gt
        assert Path(args[4]).resolve() == expected_dir.resolve()

    def test_make_prediction_saves_consecutive_ticks_without_overwriting(self, manager_deps, tmp_path, monkeypatch):
        """Distinct simulation ticks retain their own prediction payloads on disk."""
        monkeypatch.chdir(tmp_path)
        manager = CoperceptionModelManager(DummyOpt(save_npy=True, test_scenario="scen1"), "2023_01_01")
        manager.data_loader = [{"ego": {"origin_lidar": [np.zeros((2, 4))]}}]
        manager.opencood_dataset = MagicMock()
        manager.metrics_collector = MagicMock()
        first_prediction = np.ones((1, 8, 3))
        second_prediction = np.full((1, 8, 3), 2.0)
        monkeypatch.setattr(
            manager_deps["inference_utils"].inference_late_fusion,
            "side_effect",
            [(first_prediction, np.array([0.9]), first_prediction), (second_prediction, np.array([0.8]), second_prediction)],
        )
        saved_ticks = []

        def save_prediction_gt(prediction, ground_truth, point_cloud, timestamp, save_path):
            saved_ticks.append(timestamp)
            np.save(Path(save_path) / f"{timestamp:04d}_pred.npy", prediction)

        monkeypatch.setattr(manager_deps["inference_utils"], "save_prediction_gt", save_prediction_gt)
        expected_dir = tmp_path / "simulation_output/coperception/npy/scen1_2023_01_01/npy"

        manager.make_prediction(10)
        np.testing.assert_array_equal(np.load(expected_dir / "0010_pred.npy", allow_pickle=False), first_prediction)
        manager.make_prediction(11)

        assert saved_ticks == [10, 11]
        assert sorted(path.name for path in expected_dir.glob("*_pred.npy")) == ["0010_pred.npy", "0011_pred.npy"]
        np.testing.assert_array_equal(np.load(expected_dir / "0010_pred.npy", allow_pickle=False), first_prediction)
        np.testing.assert_array_equal(np.load(expected_dir / "0011_pred.npy", allow_pickle=False), second_prediction)

    @pytest.mark.parametrize("layout", ["unbatched", "list_wrapped", "batched", "v2_batch_index"])
    def test_save_prediction_retains_complete_numeric_point_cloud(self, layout, manager_deps, tmp_path, monkeypatch):
        """Saving must retain every point, independently of the collate layout."""
        monkeypatch.chdir(tmp_path)
        manager = CoperceptionModelManager(DummyOpt(save_npy=True, test_scenario="cloud"), "2023_01_01")
        points = np.arange(20, dtype=np.float32).reshape(5, 4)
        if layout == "list_wrapped":
            cloud = [points]
        elif layout == "batched":
            cloud = points[None, ...]
        elif layout == "v2_batch_index":
            manager.hypes["fusion"]["core_method"] = "IntermediateFusionDatasetV2"
            cloud = np.concatenate((np.zeros((len(points), 1), dtype=points.dtype), points), axis=1)[None, ...]
        else:
            cloud = points

        def save_prediction_gt(prediction, ground_truth, point_cloud, timestamp, save_path):
            assert point_cloud.ndim == 2
            np.save(Path(save_path) / f"{timestamp:04d}_pcd.npy", point_cloud)

        monkeypatch.setattr(manager_deps["inference_utils"], "save_prediction_gt", save_prediction_gt)
        nonempty_boxes = np.ones((1, 8, 3), dtype=np.float32)
        manager._save_prediction_npy(nonempty_boxes, nonempty_boxes, {"ego": {"origin_lidar": cloud}}, 42)

        saved = np.load(tmp_path / "simulation_output/coperception/npy/cloud_2023_01_01/npy/0042_pcd.npy", allow_pickle=False)
        assert saved.shape == (5, 4)
        assert saved.dtype == np.float32
        np.testing.assert_array_equal(saved, points)

    def test_save_prediction_reports_missing_point_cloud(self, manager_deps, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        manager = CoperceptionModelManager(DummyOpt(save_npy=True), "2023_01_01")
        with pytest.raises(ValueError, match="without an origin_lidar or lidar_np point cloud"):
            manager._save_prediction_npy(None, None, {"ego": {}}, 42)
        manager_deps["inference_utils"].save_prediction_gt.assert_not_called()

    @pytest.mark.parametrize("missing", ["prediction", "ground_truth", "both"])
    def test_missing_box_sets_save_numeric_empty_arrays_and_allow_following_frames(self, missing, manager_deps, tmp_path, monkeypatch):
        """The serializer rejects None instead of hiding it in object-dtype arrays."""

        class NumericTensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            def __getitem__(self, key):
                return NumericTensor(self.array[key])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.array

        def save_prediction_gt(prediction, ground_truth, point_cloud, timestamp, save_path):
            for suffix, tensor in (("pred.npy", prediction), ("gt.npy_test", ground_truth), ("pcd.npy", point_cloud)):
                # Match OpenCOOD's unconditional tensor conversion and file names.
                assert isinstance(tensor, NumericTensor)
                np.save(Path(save_path) / f"{timestamp:04d}_{suffix}", tensor.detach().cpu().numpy())

        monkeypatch.chdir(tmp_path)
        manager = CoperceptionModelManager(DummyOpt(save_npy=True, test_scenario="scen1"), "2023_01_01")
        manager.metrics_collector = MagicMock()
        point_cloud = NumericTensor(np.ones((3, 4), dtype=np.float32))
        manager.data_loader = [{"ego": {"origin_lidar": [point_cloud]}}]
        manager.opencood_dataset = MagicMock()
        prediction = NumericTensor(np.full((1, 8, 3), 2.0, dtype=np.float32))
        ground_truth = NumericTensor(np.full((2, 8, 3), 5.0, dtype=np.float64))
        first_prediction = None if missing in {"prediction", "both"} else prediction
        first_ground_truth = None if missing in {"ground_truth", "both"} else ground_truth
        empty_factory = MagicMock(side_effect=lambda shape: NumericTensor(np.empty(shape, dtype=np.float32)))
        monkeypatch.setattr(manager_deps["torch"], "empty", empty_factory, raising=False)
        monkeypatch.setattr(manager_deps["inference_utils"], "save_prediction_gt", save_prediction_gt)
        monkeypatch.setattr(
            manager_deps["inference_utils"].inference_late_fusion,
            "side_effect",
            [(first_prediction, None, first_ground_truth), (prediction, np.array([0.9]), ground_truth)],
        )

        manager.make_prediction(76)
        first_context = manager.metrics_collector.update.call_args.args[0]
        assert first_context["pred_box_tensor"] is first_prediction
        assert first_context["gt_box_tensor"] is first_ground_truth
        manager.make_prediction(77)

        folder = tmp_path / "simulation_output/coperception/npy/scen1_2023_01_01/npy"
        saved_pred = np.load(folder / "0076_pred.npy", allow_pickle=False)
        saved_gt = np.load(folder / "0076_gt.npy_test.npy", allow_pickle=False)
        assert saved_pred.shape == ((0, 8, 3) if first_prediction is None else (1, 8, 3))
        assert saved_gt.shape == ((0, 8, 3) if first_ground_truth is None else (2, 8, 3))
        expected_pred_dtype = ground_truth.array.dtype if missing == "prediction" else prediction.array.dtype
        assert saved_pred.dtype == expected_pred_dtype
        assert saved_gt.dtype == (ground_truth.array.dtype if first_ground_truth is not None else expected_pred_dtype)
        np.testing.assert_array_equal(np.load(folder / "0076_pcd.npy", allow_pickle=False), point_cloud.array)
        np.testing.assert_array_equal(np.load(folder / "0077_pred.npy", allow_pickle=False), prediction.array)
        np.testing.assert_array_equal(np.load(folder / "0077_gt.npy_test.npy", allow_pickle=False), ground_truth.array)
        assert len(list(folder.glob("*_pred.npy"))) == 2
        if missing == "both":
            empty_factory.assert_called_once_with((0, 8, 3))
        else:
            empty_factory.assert_not_called()

    def test_make_prediction_save_vis(self, manager_deps, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        opt = DummyOpt(save_vis=True, test_scenario="scen1")
        manager = CoperceptionModelManager(opt, "2023_01_01")
        # Ensure VoxelPostprocessor to test both 3d and bev
        manager.hypes["postprocess"]["core_method"] = "VoxelPostprocessor"
        manager.hypes["fusion"]["core_method"] = "IntermediateFusionDataset"

        manager.data_loader = [{"ego": {"origin_lidar": ["lidar"]}}]
        manager.opencood_dataset = MagicMock()
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("p", "s", "g")

        with patch.object(CoperceptionVisualizer, "render_inference_to_file") as mock_render:
            manager.make_prediction(5)

        # Verify directories
        base_dir = tmp_path / "simulation_output/coperception"
        assert (base_dir / "vis_3d/scen1_2023_01_01").exists()
        assert (base_dir / "vis_bev/scen1_2023_01_01").exists()

        assert mock_render.call_count == 2
        assert mock_render.call_args_list[0].kwargs["method"] == "3d"
        assert mock_render.call_args_list[1].kwargs["method"] == "bev"

    def test_make_prediction_show_video_vis(self, manager_deps, fake_heavy_deps):
        """Test Open3D interactions without opening windows."""
        opt = DummyOpt(show_video_vis=True)
        manager = CoperceptionModelManager(opt, "2023_01_01")

        manager.data_loader = [
            {"ego": {"origin_lidar": ["lidar1"]}},
        ]
        manager.opencood_dataset = MagicMock()
        # Ensure pred is not None
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("box", "score", "gt")

        with patch.object(
            CoperceptionVisualizer,
            "visualize_inference_sample_dataloader",
            return_value=(MagicMock(name="pcd"), {"pred": [MagicMock(name="pred_box")], "gt": [MagicMock(name="gt_box")]}),
        ) as mock_visualize:
            manager.make_prediction(0)

        # Check Visualizer creation (mocked class in conftest)
        manager_deps["Visualizer"].assert_called()
        vis_instance = manager.vis

        vis_instance.create_window.assert_called()
        assert vis_instance.add_geometry.call_count == 3
        vis_instance.update_renderer.assert_called()
        mock_visualize.assert_called_once()
        manager_deps["vis_utils"].linset_assign_list.assert_not_called()

    def test_make_prediction_warns_and_uses_first_batch_when_loader_has_multiple_batches(self, manager_deps):
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = [
            {"ego": {"origin_lidar": ["lidar_first"]}},
            {"ego": {"origin_lidar": ["lidar_second"]}},
        ]
        manager.opencood_dataset = MagicMock()

        with patch("opencda.core.common.coperception_model_manager.logger.warning") as mock_warning:
            manager.make_prediction(0)

        assert manager_deps["train_utils"].to_device.call_count == 1
        processed_batch = manager_deps["train_utils"].to_device.call_args[0][0]
        assert processed_batch["ego"]["origin_lidar"] == ["lidar_first"]
        assert mock_warning.call_count == 2
        warning_messages = [call.args[0] for call in mock_warning.call_args_list]
        assert "Expected exactly 1 batch" in warning_messages[0]
        assert warning_messages[1] == "Only the first batch will be processed."

    def test_make_prediction_warns_and_skips_when_loader_is_empty(self, manager_deps):
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = []
        manager.opencood_dataset = MagicMock()

        with patch("opencda.core.common.coperception_model_manager.logger.warning") as mock_warning:
            manager.make_prediction(0)

        manager_deps["train_utils"].to_device.assert_not_called()
        manager_deps["inference_utils"].inference_late_fusion.assert_not_called()
        assert mock_warning.call_count == 2
        assert "Expected exactly 1 batch" in mock_warning.call_args_list[0].args[0]
        assert mock_warning.call_args_list[1].args[0] == "Skipping cooperative perception prediction because the data loader is empty."


class TestCoperceptionVisualizer:
    @pytest.mark.parametrize("method", ["bev", "3d"])
    @pytest.mark.parametrize("missing", ["prediction", "ground_truth", "both", "neither"])
    def test_render_file_builds_canvas_with_missing_box_sets(self, method, missing, fake_heavy_deps, tmp_path, monkeypatch):
        """Exercise canvas assembly, rejecting None conversion while retaining clouds and available boxes."""
        import sys
        from types import ModuleType

        visualization = fake_heavy_deps["opencood"].visualization
        plotting = ModuleType("opencood.visualization.simple_plot3d")
        canvas_3d = ModuleType("opencood.visualization.simple_plot3d.canvas_3d")
        canvas_bev = ModuleType("opencood.visualization.simple_plot3d.canvas_bev")
        plotting.canvas_3d = canvas_3d
        plotting.canvas_bev = canvas_bev
        monkeypatch.setattr(visualization, "simple_plot3d", plotting, raising=False)
        for module in (plotting, canvas_3d, canvas_bev):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        canvas = MagicMock()
        canvas.canvas = np.zeros((4, 4, 3), dtype=np.uint8)
        canvas.get_canvas_coords.return_value = (np.array([[1, 2]]), np.array([True]))
        canvas_3d.Canvas_3D = MagicMock(return_value=canvas)
        canvas_bev.Canvas_BEV_heading_right = MagicMock(return_value=canvas)
        utils = fake_heavy_deps["opencood"].utils
        common_utils = ModuleType("opencood.utils.common_utils")

        def convert(tensor):
            assert tensor is not None
            return np.asarray(tensor)

        common_utils.torch_tensor_to_numpy = MagicMock(side_effect=convert)
        monkeypatch.setattr(utils, "common_utils", common_utils, raising=False)
        monkeypatch.setitem(sys.modules, common_utils.__name__, common_utils)
        module = import_module(CoperceptionVisualizer.__module__)
        save_image = MagicMock()
        monkeypatch.setattr(module.plt, "imsave", save_image)
        prediction = None if missing in {"prediction", "both"} else np.ones((1, 8, 3))
        ground_truth = None if missing in {"ground_truth", "both"} else np.zeros((2, 8, 3))
        path = tmp_path / f"{method}.png"

        CoperceptionVisualizer.render_inference_to_file(
            prediction,
            ground_truth,
            np.array([[1.0, 2.0, 0.0, 0.5]]),
            [-70, -40, -3, 70, 40, 1],
            path,
            method=method,
        )

        expected_sets = int(prediction is not None) + int(ground_truth is not None)
        assert common_utils.torch_tensor_to_numpy.call_count == expected_sets
        assert canvas.draw_boxes.call_count == expected_sets
        canvas.draw_canvas_points.assert_called_once()
        save_image.assert_called_once()
        assert save_image.call_args.args[0] == path
        assert save_image.call_args.args[1] is canvas.canvas
        factory = canvas_bev.Canvas_BEV_heading_right if method == "bev" else canvas_3d.Canvas_3D
        factory.assert_called_once()

    def test_resolve_visualization_config_merges_defaults_and_overrides(self):
        config = CoperceptionVisualizer.resolve_visualization_config(
            {
                "background": (12, 18, 24),
                "bbox_line_thickness": 7,
                "image_dpi": 600,
                "lidar_point_colors": {
                    "default": (200, 200, 200),
                    "cav-2": (10, 20, 30),
                },
                "bbox_colors": {"pred": (1, 2, 3)},
            }
        )

        assert config["background"] == (12, 18, 24)
        assert config["bbox_line_thickness"] == 7
        assert config["image_dpi"] == 600
        assert config["lidar_point_colors"]["default"] == (200, 200, 200)
        assert "ego" not in config["lidar_point_colors"]
        assert config["lidar_point_colors"]["cav-2"] == (10, 20, 30)
        assert config["bbox_colors"]["pred"] == (1, 2, 3)
        assert config["bbox_colors"]["gt"] == (0, 255, 0)

    def test_get_lidar_points_and_colors_uses_other_for_ego_when_ego_color_is_not_configured(self):
        config = CoperceptionVisualizer.resolve_visualization_config(
            {
                "lidar_point_colors": {
                    "other": (200, 200, 200),
                }
            }
        )
        batch_data = {
            "ego": {
                "origin_lidar_by_agent": [np.array([[1.0, 0.0, 0.0, 1.0]])],
                "origin_lidar_roles": ["ego"],
                "origin_lidar_agent_ids": ["cav-1"],
            }
        }

        _, colors = CoperceptionVisualizer._get_lidar_points_and_colors(batch_data, None, config)

        assert colors.tolist() == [[200, 200, 200]]

    def test_get_lidar_points_and_colors_keeps_agent_order_and_applies_id_override(self):
        config = CoperceptionVisualizer.resolve_visualization_config(
            {
                "lidar_point_colors": {
                    "default": (200, 200, 200),
                    "ego": (10, 250, 10),
                    "cav-2": (90, 80, 70),
                }
            }
        )
        batch_data = {
            "ego": {
                "origin_lidar_by_agent": [
                    np.array([[1.0, 0.0, 0.0, 1.0]]),
                    np.array([[2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]),
                ],
                "origin_lidar_roles": ["ego", "other"],
                "origin_lidar_agent_ids": ["cav-1", "cav-2"],
            }
        }

        points, colors = CoperceptionVisualizer._get_lidar_points_and_colors(batch_data, None, config)

        assert points.tolist() == [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]
        assert colors.tolist() == [[10, 250, 10], [90, 80, 70], [90, 80, 70]]

    def test_get_lidar_points_and_colors_prefers_agent_id_override_over_ego_role(self):
        config = CoperceptionVisualizer.resolve_visualization_config(
            {
                "lidar_point_colors": {
                    "default": (255, 255, 255),
                    "ego": (80, 255, 80),
                    "cav-1": (123, 45, 67),
                }
            }
        )
        batch_data = {
            "ego": {
                "origin_lidar_by_agent": [np.array([[1.0, 2.0, 3.0, 1.0]])],
                "origin_lidar_roles": ["ego"],
                "origin_lidar_agent_ids": ["cav-1"],
            }
        }

        _, colors = CoperceptionVisualizer._get_lidar_points_and_colors(batch_data, None, config)

        assert colors.tolist() == [[123, 45, 67]]

    def test_update_dataset_logs_warning_for_empty_memory_data_dict(self):
        """Empty dict as memory_data should be propagated and trigger empty-dataset warning."""
        dataset_mock = MagicMock()
        dataset_mock.collate_batch_test = MagicMock()
        dataset_mock.__len__.return_value = 0

        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock):
            manager = CoperceptionModelManager(DummyOpt(), "2023_01_01")

        with patch("opencda.core.common.coperception_model_manager.logger.warning") as mock_warning:
            manager.update_dataset(data={})

        dataset_mock.update_database.assert_called_once_with(memory_data={})
        mock_warning.assert_called_once_with("No samples found in dataset after update.")

    def test_update_dataset_logs_warning_for_none_memory_data(self):
        """None as memory_data should be propagated and trigger empty-dataset warning."""
        dataset_mock = MagicMock()
        dataset_mock.collate_batch_test = MagicMock()
        dataset_mock.__len__.return_value = 0

        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock):
            manager = CoperceptionModelManager(DummyOpt(), "2023_01_01")

        with patch("opencda.core.common.coperception_model_manager.logger.warning") as mock_warning:
            manager.update_dataset(data=None)

        dataset_mock.update_database.assert_called_once_with(memory_data=None)
        mock_warning.assert_called_once_with("No samples found in dataset after update.")


class TestCoperceptionDataProcessor:
    _DEFAULT_MEASUREMENT_DATA = object()

    @staticmethod
    def _make_transform(x=0.0, y=0.0, z=0.0, roll=0.0, yaw=0.0, pitch=0.0):
        return MagicMock(
            location=MagicMock(x=x, y=y, z=z),
            rotation=MagicMock(roll=roll, yaw=yaw, pitch=pitch),
        )

    @staticmethod
    def _make_bounding_box(x=0.0, y=0.0, z=0.0, ex=1.0, ey=2.0, ez=3.0):
        return MagicMock(
            location=MagicMock(x=x, y=y, z=z),
            extent=MagicMock(x=ex, y=ey, z=ez),
        )

    @staticmethod
    def _make_measurement(frame=12, data=_DEFAULT_MEASUREMENT_DATA, transform=None):
        return MagicMock(
            frame=frame,
            timestamp=float(frame),
            transform=transform,
            data=np.array([[1.0, 2.0, 3.0, 1.0]], dtype=np.float32) if data is TestCoperceptionDataProcessor._DEFAULT_MEASUREMENT_DATA else data,
        )

    def test_build_live_camera_snapshots_returns_placeholder_list(self):
        processor = CoperceptionDataProcessor()

        assert processor._build_live_camera_snapshots(MagicMock()) == []

    def test_build_live_params_with_vehicle_localizer_and_behavior_agent(self):
        processor = CoperceptionDataProcessor()

        valid_vehicle = MagicMock()
        valid_vehicle.carla_id = 42
        valid_vehicle.type_id = "vehicle.tesla.model3"
        valid_vehicle.color = "255,0,0"
        valid_vehicle.bounding_box = self._make_bounding_box(0.1, 0.2, 0.3, 1.1, 1.2, 1.3)
        valid_vehicle.get_transform.return_value = self._make_transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        ignored_vehicle = MagicMock()
        ignored_vehicle.carla_id = -1

        lidar_transform = self._make_transform(10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
        camera_transform = self._make_transform(11.0, 21.0, 31.0, 4.0, 5.0, 6.0)
        perception_manager = MagicMock()
        perception_manager.objects = {"vehicles": [valid_vehicle, ignored_vehicle]}
        perception_manager.lidar = MagicMock(sensor=MagicMock(get_transform=MagicMock(return_value=lidar_transform)))
        perception_manager.rgb_camera = [MagicMock(sensor=MagicMock(get_transform=MagicMock(return_value=camera_transform)))]

        predicted_ego_pos = self._make_transform(100.0, 200.0, 300.0, 7.0, 8.0, 9.0)
        true_ego_pos = self._make_transform(101.0, 201.0, 301.0, 10.0, 11.0, 12.0)
        localizer = MagicMock()
        localizer.get_state.return_value = MagicMock(transform=predicted_ego_pos, speed_kmh=13.5)
        actor = MagicMock(get_transform=MagicMock(return_value=true_ego_pos))

        waypoint = MagicMock(location=MagicMock(x=1.5, y=2.5))
        behavior_agent = MagicMock()
        behavior_agent.get_local_planner.return_value.get_trajectory.return_value = [(waypoint, 6.7)]

        with (
            patch("opencda.core.common.coperception_data_processor.get_speed", return_value=22.2),
            patch(
                "opencda.core.common.coperception_data_processor.st.get_camera_intrinsic",
                return_value=np.eye(3),
            ),
            patch(
                "opencda.core.common.coperception_data_processor.st.x_to_world_transformation",
                side_effect=[np.eye(4), np.eye(4)],
            ),
        ):
            params = processor.build_live_params(
                perception_manager,
                localizer,
                actor,
                behavior_agent,
                {
                    "lidar": self._make_measurement(transform=lidar_transform),
                    "camera0": self._make_measurement(transform=camera_transform),
                },
                is_rsu=False,
            )

        assert params["RSU"] is False
        assert params["ego_speed"] == 13.5
        assert params["predicted_ego_pos"] == (100.0, 200.0, 300.0, 7.0, 8.0, 9.0)
        assert params["true_ego_pos"] == (101.0, 201.0, 301.0, 10.0, 11.0, 12.0)
        assert params["lidar_pose"] == (10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
        assert params["plan_trajectory"] == [(1.5, 2.5, 6.7)]
        assert list(params["vehicles"].keys()) == [42]
        assert params["vehicles"][42]["speed"] == 22.2
        assert params["camera0"]["cords"] == (11.0, 21.0, 31.0, 4.0, 5.0, 6.0)
        assert params["camera0"]["intrinsic"] == np.eye(3).tolist()
        assert params["camera0"]["extrinsic"] == np.eye(4).tolist()

    def test_build_live_params_with_rsu_localizer_sets_rsu_true(self):
        processor = CoperceptionDataProcessor()

        lidar_transform = self._make_transform(10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
        perception_manager = MagicMock()
        perception_manager.objects = {"vehicles": []}
        perception_manager.lidar = MagicMock(sensor=MagicMock(get_transform=MagicMock(return_value=lidar_transform)))
        perception_manager.rgb_camera = []

        predicted_ego_pos = self._make_transform(50.0, 60.0, 70.0, 0.0, 90.0, 0.0)
        true_ego_pos = self._make_transform(51.0, 61.0, 71.0, 0.0, 91.0, 0.0)
        localizer = MagicMock()
        localizer.get_state.return_value = MagicMock(transform=predicted_ego_pos, speed_kmh=0.0)
        actor = MagicMock(get_transform=MagicMock(return_value=true_ego_pos))

        params = processor.build_live_params(
            perception_manager,
            localizer,
            actor,
            None,
            {"lidar": self._make_measurement(transform=lidar_transform)},
            is_rsu=True,
        )

        assert params["RSU"] is True
        assert "plan_trajectory" not in params
        assert params["true_ego_pos"] == (51.0, 61.0, 71.0, 0.0, 91.0, 0.0)

    def test_build_live_memory_returns_none_and_warns_for_empty_agents(self):
        processor = CoperceptionDataProcessor()

        with patch("opencda.core.common.coperception_data_processor.logger.warning") as mock_warning:
            memory = processor.build_live_memory([], [], 5, sensor_frame=12)

        assert memory is None
        mock_warning.assert_called_once()
        assert mock_warning.call_args[0] == (
            "Skipping cooperative perception tick %s because there are no CAV or RSU agents.",
            5,
        )

    def test_build_live_memory_marks_first_vehicle_as_ego_and_formats_timestamp(self):
        processor = CoperceptionDataProcessor()

        cav1 = MagicMock()
        cav1.id = "cav-1"
        cav1.agent = MagicMock()
        cav1.agent.is_vehicle = True
        cav1.agent.perception_manager = MagicMock(lidar=MagicMock(data=np.array([[1.0, 2.0, 3.0, 1.0]])))

        cav2 = MagicMock()
        cav2.id = "cav-2"
        cav2.agent = MagicMock()
        cav2.agent.is_vehicle = True
        cav2.agent.perception_manager = MagicMock(lidar=MagicMock(data=np.array([[4.0, 5.0, 6.0, 1.0]])))

        rsu = MagicMock()
        rsu.id = "rsu-1"
        rsu.agent = MagicMock()
        rsu.agent.is_vehicle = False
        rsu.agent.perception_manager = MagicMock(lidar=MagicMock(data=np.array([[7.0, 8.0, 9.0, 1.0]])))

        with (
            patch.object(
                CoperceptionDataProcessor,
                "_wait_for_sensor_frame",
                side_effect=[
                    {"lidar": self._make_measurement(data=np.array([[1.0, 2.0, 3.0, 1.0]], dtype=np.float32))},
                    {"lidar": self._make_measurement(data=np.array([[4.0, 5.0, 6.0, 1.0]], dtype=np.float32))},
                    {"lidar": self._make_measurement(data=np.array([[7.0, 8.0, 9.0, 1.0]], dtype=np.float32))},
                ],
            ),
            patch.object(CoperceptionDataProcessor, "build_live_params", side_effect=[{"id": 1}, {"id": 2}, {"id": 3}]),
            patch.object(
                CoperceptionDataProcessor,
                "_build_live_camera_snapshots",
                return_value=[],
            ),
        ):
            memory = processor.build_live_memory([cav1, cav2], [rsu], 12, sensor_frame=12)

        assert list(memory.keys()) == [0]
        assert list(memory[0].keys()) == ["cav-1", "cav-2", "rsu-1"]
        assert memory[0]["cav-1"]["ego"] is True
        assert memory[0]["cav-2"]["ego"] is False
        assert memory[0]["rsu-1"]["ego"] is False
        assert memory[0]["cav-1"]["000012"]["params"] == {"id": 1}
        assert memory[0]["cav-2"]["000012"]["params"] == {"id": 2}
        assert memory[0]["rsu-1"]["000012"]["params"] == {"id": 3}
        assert memory[0]["cav-1"]["000012"]["lidar_np"].tolist() == [[1.0, 2.0, 3.0, 1.0]]
        assert memory[0]["rsu-1"]["000012"]["lidar_np"].tolist() == [[7.0, 8.0, 9.0, 1.0]]
        assert memory[0]["cav-1"]["000012"]["camera0"] == []

    def test_build_live_memory_skips_agent_when_vehicle_lidar_is_missing(self):
        processor = CoperceptionDataProcessor()
        cav = MagicMock()
        cav.id = "cav-1"
        cav.agent = MagicMock()
        cav.agent.perception_manager = MagicMock(lidar=None)

        with patch("opencda.core.common.coperception_data_processor.logger.warning") as mock_warning:
            memory = processor.build_live_memory([cav], [], 1, sensor_frame=12)

        assert memory is None
        assert mock_warning.call_count == 2
        assert "LiDAR is not initialized" in mock_warning.call_args_list[0].args[0]
        assert "no agents have valid LiDAR data" in mock_warning.call_args_list[1].args[0]

    def test_build_live_memory_skips_agent_when_vehicle_lidar_data_is_missing(self):
        processor = CoperceptionDataProcessor()
        cav = MagicMock()
        cav.id = "cav-1"
        cav.agent = MagicMock()
        cav.agent.perception_manager = MagicMock(lidar=MagicMock(data=None))

        with (
            patch.object(
                CoperceptionDataProcessor,
                "_wait_for_sensor_frame",
                return_value={"lidar": self._make_measurement(data=None)},
            ),
            patch("opencda.core.common.coperception_data_processor.logger.warning") as mock_warning,
        ):
            memory = processor.build_live_memory([cav], [], 1, sensor_frame=12)

        assert memory is None
        assert mock_warning.call_count == 2
        assert "LiDAR data is not initialized" in mock_warning.call_args_list[0].args[0]
        assert "no agents have valid LiDAR data" in mock_warning.call_args_list[1].args[0]
