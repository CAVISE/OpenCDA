import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# The production code imports are now safe because pytest_configure in conftest.py
# installs the mocks before collection.
from opencda.core.common.coperception_model_manager import CoperceptionModelManager, DirectoryProcessor


class DummyOpt:
    def __init__(self, **kwargs):
        self.model_dir = "test_model_dir"
        self.fusion_method = "late"
        self.show_vis = False
        self.show_sequence = False
        self.save_npy = False
        self.save_vis = False
        self.test_scenario = "test_scenario"
        self.global_sort_detections = True
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

    def update_database(self):
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
            opencood.visualization.simple_vis.visualize,
            opencood.visualization.vis_utils.visualize_inference_sample_dataloader,
            opencood.visualization.vis_utils.linset_assign_list,
            opencood.utils.eval_utils.caluclate_tp_fp,
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
            "fusion": {"core_method": "IntermediateFusionDataset"},
        }
        opencood.hypes_yaml.yaml_utils.load_yaml.return_value = hypes

        model = MagicMock()
        opencood.tools.train_utils.create_model.return_value = model
        opencood.tools.train_utils.load_saved_model.return_value = (None, model)

        # Return dict for easy access
        return {
            "yaml_utils": opencood.hypes_yaml.yaml_utils,
            "train_utils": opencood.tools.train_utils,
            "inference_utils": opencood.tools.inference_utils,
            "vis_utils": opencood.visualization.vis_utils,
            "simple_vis": opencood.visualization.simple_vis,
            "eval_utils": opencood.utils.eval_utils,
            "build_dataset": opencood.data_utils.datasets.build_dataset,
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

    def test_update_dataset(self, manager_deps):
        """
        Verify update_dataset calls the correct build_dataset and creates a DataLoader.
        We patch the symbol inside the module under test to ensure we capture the call.
        """
        dataset_mock = DummyDataset()

        # Patching where it is imported in the source code
        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock) as mock_build:
            opt = DummyOpt()
            manager = CoperceptionModelManager(opt, "2023_01_01")

            manager.update_dataset()

            mock_build.assert_called_with(manager_deps["hypes"], visualize=True, train=False, message_handler=None)
            assert manager.opencood_dataset == dataset_mock
            assert manager.data_loader is not None
            assert manager.data_loader.dataset == dataset_mock

    def test_make_prediction_state_update(self, manager_deps):
        """Test that final_result_stat is actually updated via caluclate_tp_fp side effect."""
        opt = DummyOpt(fusion_method="late")
        manager = CoperceptionModelManager(opt, "2023_01_01")

        # Setup Data Loader
        manager.data_loader = [{"ego": {"origin_lidar": ["lidar_data"]}}]
        manager.opencood_dataset = MagicMock()

        # Define side effect for caluclate_tp_fp to modify the stats dictionary
        def mock_calculate_tp_fp(pred, score, gt, stat, iou):
            stat[iou]["gt"] += 1
            stat[iou]["tp"].append(1)
            stat[iou]["fp"].append(0)
            stat[iou]["score"].append(0.9)

        manager_deps["eval_utils"].caluclate_tp_fp.side_effect = mock_calculate_tp_fp

        manager.make_prediction(0)

        # Verify stats were accumulated
        for iou in [0.3, 0.5, 0.7]:
            assert manager.final_result_stat[iou]["gt"] == 1
            assert len(manager.final_result_stat[iou]["tp"]) == 1
            assert manager.final_result_stat[iou]["score"][0] == 0.9

    @pytest.mark.parametrize("fusion_method", ["late", "early", "intermediate"])
    def test_make_prediction_fusion_methods(self, fusion_method, manager_deps):
        opt = DummyOpt(fusion_method=fusion_method)
        manager = CoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = [{"ego": {"origin_lidar": ["lidar_data"]}}]
        manager.opencood_dataset = MagicMock()

        manager.make_prediction(0)

        if fusion_method == "late":
            manager_deps["inference_utils"].inference_late_fusion.assert_called()
        elif fusion_method == "early":
            manager_deps["inference_utils"].inference_early_fusion.assert_called()
        elif fusion_method == "intermediate":
            manager_deps["inference_utils"].inference_intermediate_fusion.assert_called()

    def test_make_prediction_assertions(self):
        opt = DummyOpt(fusion_method="invalid")
        manager = CoperceptionModelManager(opt, "2023_01_01")
        with pytest.raises(AssertionError):
            manager.make_prediction(0)

        opt = DummyOpt(fusion_method="late", show_vis=True, show_sequence=True)
        manager = CoperceptionModelManager(opt, "2023_01_01")
        with pytest.raises(AssertionError, match="single image mode or video mode"):
            manager.make_prediction(0)

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
        # args[4] is the path passed to save_prediction_gt
        assert Path(args[4]).resolve() == expected_dir.resolve()

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

        manager.make_prediction(5)

        # Verify directories
        base_dir = tmp_path / "simulation_output/coperception"
        assert (base_dir / "vis_3d/scen1_2023_01_01").exists()
        assert (base_dir / "vis_bev/scen1_2023_01_01").exists()

        assert manager_deps["simple_vis"].visualize.call_count == 2

    def test_make_prediction_show_sequence(self, manager_deps, fake_heavy_deps):
        """Test Open3D interactions without opening windows."""
        opt = DummyOpt(show_sequence=True)
        manager = CoperceptionModelManager(opt, "2023_01_01")

        manager.data_loader = [
            {"ego": {"origin_lidar": ["lidar1"]}},
        ]
        manager.opencood_dataset = MagicMock()
        # Ensure pred is not None
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("box", "score", "gt")

        manager.make_prediction(0)

        # Check Visualizer creation (mocked class in conftest)
        manager_deps["Visualizer"].assert_called()
        vis_instance = manager.vis

        vis_instance.create_window.assert_called()
        vis_instance.add_geometry.assert_called()  # i=0
        vis_instance.update_renderer.assert_called()

        # Verify line set assignment was called
        manager_deps["vis_utils"].linset_assign_list.assert_called()

    def test_final_eval(self, manager_deps, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        opt = DummyOpt(test_scenario="scen1")
        manager = CoperceptionModelManager(opt, "2023_01_01")

        manager.final_eval()

        expected_dir = tmp_path / "simulation_output/coperception/results/scen1_2023_01_01"
        assert expected_dir.is_dir()

        manager_deps["eval_utils"].eval_final_results.assert_called()
        args = manager_deps["eval_utils"].eval_final_results.call_args[0]
        # Check path arg
        assert args[0] is manager.final_result_stat
        assert Path(args[1]).resolve() == expected_dir.resolve()
        assert args[2] == opt.global_sort_detections


# --- Tests for DirectoryProcessor ---


class TestDirectoryProcessor:
    @pytest.fixture
    def processor_setup(self, tmp_path):
        source_dir = tmp_path / "data_dumping"
        # IMPORTANT: now_dir must NOT be inside source_dir, otherwise it becomes part of
        # subdirectories and breaks the "subdirectories[-2]" selection logic.
        now_dir = tmp_path / "now"
        source_dir.mkdir(parents=True)
        now_dir.mkdir(parents=True)
        return source_dir, now_dir

    def test_detect_cameras(self, tmp_path):
        dp = DirectoryProcessor()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        assert dp.detect_cameras(str(data_dir)) == []

        sample = data_dir / "sample_subdir"
        sample.mkdir()
        (sample / "001_camera0.png").touch()
        (sample / "001_camera2.png").touch()
        (sample / "001_camera1.png").touch()

        cameras = dp.detect_cameras(str(data_dir))
        assert cameras == ["_camera0.png", "_camera1.png", "_camera2.png"]

    def test_process_directory_success(self, processor_setup):
        source_dir, now_dir = processor_setup
        dp = DirectoryProcessor(str(source_dir), str(now_dir))

        # Needs at least 2 dirs. Sorted order: d1, d2.
        # Code picks index -2 -> d1.
        d1 = source_dir / "d1"
        d2 = source_dir / "d2"
        d1.mkdir()
        d2.mkdir()

        (d1 / "data_protocol.yaml").write_text("proto")
        agent1 = d1 / "agent1"
        agent1.mkdir()

        # Files for tick 10
        (agent1 / "000010.pcd").write_text("pcd")
        (agent1 / "000010.yaml").write_text("yaml")

        dp.process_directory(10)

        assert (now_dir / "data_protocol.yaml").exists()
        assert (now_dir / "data_protocol.yaml").read_text() == "proto"
        assert (now_dir / "agent1" / "000010.pcd").exists()
        assert (now_dir / "agent1" / "000010.pcd").read_text() == "pcd"

    def test_clear_directory_now(self, processor_setup):
        _, now_dir = processor_setup
        dp = DirectoryProcessor(now_directory=str(now_dir))
        (now_dir / "file.txt").touch()
        (now_dir / "subdir").mkdir()

        dp.clear_directory_now()

        assert len(os.listdir(now_dir)) == 0
