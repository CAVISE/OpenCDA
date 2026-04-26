import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

# The production code imports are now safe because pytest_configure in conftest.py
# installs the mocks before collection.
from opencda.core.attack.advcp import AdvCoperceptionModelManager
from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.adv_coperception_model_manager import AdvCoperceptionVisualizer
from opencda.core.attack.advcp.early_fusion_attack import AdvCoperceptionEarlyFusionAttack
from opencda.core.attack.advcp.intermediate_fusion_attack import AdvCoperceptionIntermediateFusionAttack
from opencda.core.attack.advcp.late_fusion_attack import AdvCoperceptionLateFusionAttack
from opencda.core.common.coperception_data_processor import CoperceptionDataProcessor
from opencda.core.common.coperception_model_manager import (
    CoperceptionInferenceResult,
    CoperceptionModelManager,
    CoperceptionVisualizer,
    EvaluationResultStat,
    IoUResultStat,
)


class DummyOpt:
    def __init__(self, **kwargs):
        self.model_dir = "test_model_dir"
        self.fusion_method = "late"
        self.show_video_vis = False
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

    def update_database(self, memory_data=None):
        pass


def make_advcp_config(**overrides):
    config = {
        "mode": "spoof",
        "attacker_ids": ["cav-2"],
        "boxes": [{"relative": (5.0, 0.0, 0.0, 0.0, 90.0, 0.0)}],
        "default_size": (4.5, 2.0, 1.6),
        "density": 3,
        "dense_distance": 10.0,
        "sync": True,
        "init": True,
        "online": True,
        "step": 25,
        "random_seed": 1,
        "max_perturb": 10.0,
        "lr": 0.05,
        "feature_size": 10,
        "car_mesh_path": "dummy_car_mesh.ply",
        "car_mesh_divide_path": "dummy_car_mesh_divide.pkl",
    }
    config.update(overrides)
    return config


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
            "fusion": {"core_method": "LateFusionDataset"},
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
        dataset_mock.update_database = MagicMock()
        memory_data = {0: {"cav_1": {"000010": {"yaml": "a.yaml", "lidar": "a.pcd"}}}}

        # Patching where it is imported in the source code
        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock) as mock_build:
            opt = DummyOpt()
            manager = CoperceptionModelManager(opt, "2023_01_01")

            manager.update_dataset(data=memory_data)

            dataset_mock.update_database.assert_called_once_with(memory_data=memory_data)
            mock_build.assert_called_with(manager_deps["hypes"], visualize=True, train=False, payload_handler=None)
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

    def test_make_prediction_state_update(self, manager_deps):
        """Test that final_result_stat is actually updated via caluclate_tp_fp side effect."""
        opt = DummyOpt()
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

    def test_evaluation_result_stat_merges_nested_iou_stats(self):
        accumulated = EvaluationResultStat.create_empty()
        batch = EvaluationResultStat.create_empty()

        batch[0.3]["gt"] = 2
        batch[0.3]["tp"].extend([1, 1])
        batch[0.3]["fp"].append(0)
        batch[0.3]["score"].extend([0.7, 0.8])

        batch[0.5]["gt"] = 1
        batch[0.5]["tp"].append(1)
        batch[0.5]["score"].append(0.9)

        accumulated.merge_from(batch)

        assert accumulated[0.3]["gt"] == 2
        assert accumulated[0.3]["tp"] == [1, 1]
        assert accumulated[0.3]["fp"] == [0]
        assert accumulated[0.3]["score"] == [0.7, 0.8]
        assert accumulated[0.5]["gt"] == 1
        assert accumulated[0.5]["tp"] == [1]
        assert accumulated[0.5]["score"] == [0.9]

    def test_iou_result_stat_supports_dict_style_updates(self):
        stat = IoUResultStat.create_empty()

        stat["gt"] += 1
        stat["tp"].append(1)
        stat["fp"].append(0)
        stat["score"].append(0.5)

        assert stat.gt == 1
        assert stat.tp == [1]
        assert stat.fp == [0]
        assert stat.score == [0.5]

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

    def test_make_prediction_late_advcp_dispatches_to_late_attack_class(self, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = "LateFusionDataset"
        opt = DummyOpt(with_advcp=True, advcp_config="dummy.yaml")
        with patch.object(AdvCoperceptionModelManager, "load_config", return_value=make_advcp_config(boxes=[{}])):
            manager = AdvCoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = [{"ego": {"origin_lidar": ["lidar_data"]}}]
        manager.opencood_dataset = MagicMock()

        with patch.object(
            AdvCoperceptionLateFusionAttack,
            "run",
            return_value=("pred", "score", "gt", {"attacker_ids": ["cav-2"], "fake_box_tensor": "fake"}),
        ) as mock_advcp:
            manager.make_prediction(0)
            result = manager.inference({"ego": {"origin_lidar": ["lidar_data"]}})

        assert mock_advcp.call_count == 2
        assert isinstance(result, CoperceptionInferenceResult)
        assert result.visualization_context == {"attacker_ids": ["cav-2"], "fake_box_tensor": "fake"}

    def test_make_prediction_early_advcp_dispatches_to_early_attack_class(self, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = "EarlyFusionDataset"
        opt = DummyOpt(with_advcp=True, advcp_config="dummy.yaml")
        with patch.object(AdvCoperceptionModelManager, "load_config", return_value=make_advcp_config(boxes=[{}])):
            manager = AdvCoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = [{"ego": {"origin_lidar": ["lidar_data"]}}]
        manager.opencood_dataset = MagicMock()

        with patch.object(
            AdvCoperceptionEarlyFusionAttack,
            "run",
            return_value=("pred", "score", "gt", {"attacker_ids": ["cav-2"], "fake_box_tensor": None, "mode": "spoof"}),
        ) as mock_attack:
            manager.make_prediction(0)
            result = manager.inference({"ego": {"origin_lidar": ["lidar_data"]}})

        assert mock_attack.call_count == 2
        assert isinstance(result, CoperceptionInferenceResult)
        assert result.visualization_context == {"attacker_ids": ["cav-2"], "fake_box_tensor": None, "mode": "spoof"}

    def test_make_prediction_intermediate_advcp_dispatches_to_intermediate_attack_class(self, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = "IntermediateFusionDataset"
        opt = DummyOpt(with_advcp=True, advcp_config="dummy.yaml")
        with patch.object(AdvCoperceptionModelManager, "load_config", return_value=make_advcp_config(boxes=[{}])):
            manager = AdvCoperceptionModelManager(opt, "2023_01_01")
        manager.data_loader = [{"ego": {"origin_lidar": ["lidar_data"]}}]
        manager.opencood_dataset = MagicMock()

        with patch.object(
            AdvCoperceptionIntermediateFusionAttack,
            "run",
            return_value=("pred", "score", "gt", {"attacker_ids": ["cav-2"], "fake_box_tensor": None, "mode": "spoof"}),
        ) as mock_attack:
            manager.make_prediction(0)
            result = manager.inference({"ego": {"origin_lidar": ["lidar_data"]}})

        assert mock_attack.call_count == 2
        assert isinstance(result, CoperceptionInferenceResult)
        assert result.visualization_context == {"attacker_ids": ["cav-2"], "fake_box_tensor": None, "mode": "spoof"}

    def test_intermediate_advcp_requires_grad_for_inference(self, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = "IntermediateFusionDataset"
        opt = DummyOpt(with_advcp=True, advcp_config="dummy.yaml")
        with patch.object(AdvCoperceptionModelManager, "load_config", return_value=make_advcp_config(boxes=[{}])):
            manager = AdvCoperceptionModelManager(opt, "2023_01_01")

        assert manager._requires_grad_for_inference() is True

    def test_intermediate_advcp_passes_attack_state_to_handler(self, manager_deps):
        manager_deps["hypes"]["fusion"]["core_method"] = "IntermediateFusionDataset"
        opt = DummyOpt(with_advcp=True, advcp_config="dummy.yaml")
        with patch.object(AdvCoperceptionModelManager, "load_config", return_value=make_advcp_config(boxes=[{}])):
            manager = AdvCoperceptionModelManager(opt, "2023_01_01")
        manager.opencood_dataset = MagicMock()

        with patch.object(
            AdvCoperceptionIntermediateFusionAttack,
            "run",
            return_value=("pred", "score", "gt", {"attacker_ids": ["cav-2"], "fake_box_tensor": None, "mode": "spoof"}),
        ) as mock_attack:
            manager.inference({"ego": {"origin_lidar": ["lidar_data"]}})

        assert mock_attack.call_args.kwargs["attack_state"] is manager.intermediate_attack_state

    def test_early_advcp_run_falls_back_when_attacker_is_missing(self, manager_deps):
        batch_data = {"ego": {"origin_lidar": ["lidar_data"]}}
        dataset = MagicMock()
        model = MagicMock()
        memory_data = {
            0: {
                "cav-1": {
                    "ego": True,
                    "000001": {
                        "params": {
                            "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        }
                    },
                }
            }
        }

        result = AdvCoperceptionEarlyFusionAttack.run(
            batch_data,
            model,
            dataset,
            "device(cpu)",
            make_advcp_config(),
            memory_data=memory_data,
        )

        manager_deps["inference_utils"].inference_early_fusion.assert_called_once_with(batch_data, model, dataset)
        assert result[0] == manager_deps["inference_utils"].inference_early_fusion.return_value[0]
        assert result[3] == {"attacker_ids": [], "fake_box_tensor": None, "mode": "spoof"}

    def test_intermediate_advcp_run_falls_back_when_attacker_is_missing(self, manager_deps):
        batch_data = {"ego": {"origin_lidar": ["lidar_data"]}}
        dataset = MagicMock()
        model = MagicMock()
        memory_data = {
            0: {
                "cav-1": {
                    "ego": True,
                    "000001": {
                        "params": {
                            "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        }
                    },
                }
            }
        }

        result = AdvCoperceptionIntermediateFusionAttack.run(
            batch_data,
            model,
            dataset,
            "device(cpu)",
            make_advcp_config(),
            memory_data=memory_data,
        )

        manager_deps["inference_utils"].inference_intermediate_fusion.assert_called_once_with(batch_data, model, dataset)
        assert result[0] == manager_deps["inference_utils"].inference_intermediate_fusion.return_value[0]
        assert result[3] == {"attacker_ids": [], "fake_box_tensor": None, "mode": "spoof"}

    def test_intermediate_advcp_run_falls_back_when_attacker_is_missing_from_batch(self, manager_deps):
        batch_data = {
            "ego": {
                "origin_lidar_agent_ids": ["cav-1"],
            }
        }
        dataset = MagicMock()
        model = MagicMock()
        memory_data = {
            0: {
                "cav-1": {"ego": True, "000001": {"params": {"lidar_pose": [0.0] * 6, "true_ego_pos": [0.0] * 6}}},
                "cav-2": {"ego": False, "000001": {"params": {"lidar_pose": [1.0] * 6, "true_ego_pos": [1.0] * 6}}},
            }
        }

        with patch.object(
            AdvCPAttackHelper,
            "resolve_spoof_boxes_for_agent",
            return_value=("cav-1", MagicMock(), MagicMock(), [np.array([5.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0], dtype=np.float32)]),
        ):
            result = AdvCoperceptionIntermediateFusionAttack.run(
                batch_data,
                model,
                dataset,
                "device(cpu)",
                make_advcp_config(),
                memory_data=memory_data,
            )

        manager_deps["inference_utils"].inference_intermediate_fusion.assert_called_once_with(batch_data, model, dataset)
        assert result[0] == manager_deps["inference_utils"].inference_intermediate_fusion.return_value[0]
        assert result[3] == {"attacker_ids": [], "fake_box_tensor": None, "mode": "spoof"}

    def test_early_advcp_run_rebuilds_batch_with_attacked_memory(self, manager_deps):
        batch_data = {"old": "batch"}
        dataset = MagicMock()
        dataset.__getitem__.return_value = {"ego": {"processed_lidar": "item"}}
        dataset.collate_batch_test.return_value = {"rebuilt": "batch"}
        model = MagicMock()
        memory_data = {
            0: {
                "cav-1": {
                    "ego": True,
                    "000001": {
                        "params": {
                            "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        },
                        "lidar_np": np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                    },
                },
                "cav-2": {
                    "ego": False,
                    "000001": {
                        "params": {
                            "lidar_pose": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        },
                        "lidar_np": np.array([[2.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                    },
                },
            }
        }
        states = {
            "cav-1": {
                "agent_id": "cav-1",
                "params": memory_data[0]["cav-1"]["000001"]["params"],
                "lidar_pose": memory_data[0]["cav-1"]["000001"]["params"]["lidar_pose"],
                "ego_pose": memory_data[0]["cav-1"]["000001"]["params"]["true_ego_pos"],
            },
            "cav-2": {
                "agent_id": "cav-2",
                "params": memory_data[0]["cav-2"]["000001"]["params"],
                "lidar_pose": memory_data[0]["cav-2"]["000001"]["params"]["lidar_pose"],
                "ego_pose": memory_data[0]["cav-2"]["000001"]["params"]["true_ego_pos"],
            },
        }
        spoof_box = np.array([5.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0], dtype=np.float32)
        spoofed_lidar = np.array([[9.0, 9.0, 9.0, 1.0]], dtype=np.float32)

        with (
            patch.object(AdvCPAttackHelper, "load_agent_state", side_effect=lambda scenario_data, agent_id: states[agent_id]),
            patch.object(
                AdvCPAttackHelper,
                "resolve_spoof_boxes_for_agent",
                return_value=("cav-1", states["cav-1"], states["cav-2"], [spoof_box]),
            ),
            patch.object(
                AdvCoperceptionEarlyFusionAttack,
                "_apply_sampled_ray_traced_spoof",
                return_value=(spoofed_lidar, np.array([True])),
            ),
        ):
            result = AdvCoperceptionEarlyFusionAttack.run(
                batch_data,
                model,
                dataset,
                "device(cpu)",
                make_advcp_config(),
                memory_data=memory_data,
            )

        first_update = dataset.update_database.call_args_list[0].kwargs["memory_data"]
        restored_update = dataset.update_database.call_args_list[1].kwargs["memory_data"]

        assert first_update is not memory_data
        np.testing.assert_array_equal(first_update[0]["cav-2"]["000001"]["lidar_np"], spoofed_lidar)
        np.testing.assert_array_equal(first_update[0]["cav-2"]["000001"]["spoofing_mask"], np.array([True]))
        assert restored_update is memory_data
        assert batch_data == {"rebuilt": "batch"}
        assert result[3] == {"attacker_ids": ["cav-2"], "fake_box_tensor": None, "mode": "spoof"}
        manager_deps["inference_utils"].inference_early_fusion.assert_called_once_with({"rebuilt": "batch"}, model, dataset)

    def test_early_advcp_run_supports_multiple_attackers(self, manager_deps):
        batch_data = {"old": "batch"}
        dataset = MagicMock()
        dataset.__getitem__.return_value = {"ego": {"processed_lidar": "item"}}
        dataset.collate_batch_test.return_value = {"rebuilt": "batch"}
        model = MagicMock()
        memory_data = {
            0: {
                "cav-1": {
                    "ego": True,
                    "000001": {
                        "params": {
                            "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        },
                        "lidar_np": np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                    },
                },
                "cav-2": {
                    "ego": False,
                    "000001": {
                        "params": {
                            "lidar_pose": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        },
                        "lidar_np": np.array([[2.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                    },
                },
                "cav-3": {
                    "ego": False,
                    "000001": {
                        "params": {
                            "lidar_pose": [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "true_ego_pos": [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        },
                        "lidar_np": np.array([[3.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                    },
                },
            }
        }
        states = {
            "cav-1": {
                "agent_id": "cav-1",
                "params": memory_data[0]["cav-1"]["000001"]["params"],
                "lidar_pose": memory_data[0]["cav-1"]["000001"]["params"]["lidar_pose"],
                "ego_pose": memory_data[0]["cav-1"]["000001"]["params"]["true_ego_pos"],
            },
            "cav-2": {
                "agent_id": "cav-2",
                "params": memory_data[0]["cav-2"]["000001"]["params"],
                "lidar_pose": memory_data[0]["cav-2"]["000001"]["params"]["lidar_pose"],
                "ego_pose": memory_data[0]["cav-2"]["000001"]["params"]["true_ego_pos"],
            },
            "cav-3": {
                "agent_id": "cav-3",
                "params": memory_data[0]["cav-3"]["000001"]["params"],
                "lidar_pose": memory_data[0]["cav-3"]["000001"]["params"]["lidar_pose"],
                "ego_pose": memory_data[0]["cav-3"]["000001"]["params"]["true_ego_pos"],
            },
        }
        spoof_box = np.array([5.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0], dtype=np.float32)

        def _mock_resolve_spoof_boxes_for_agent(_scenario_data, _advcp_config, attacker_id):
            return ("cav-1", states["cav-1"], states[attacker_id], [spoof_box])

        def _mock_apply_sampled_ray_traced_spoof(_lidar, _spoofing_mask, _attack_box, _lidar_poses, attacker_id, *_):
            spoof_point = 9.0 if attacker_id == "cav-2" else 8.0
            return np.array([[spoof_point, spoof_point, spoof_point, 1.0]], dtype=np.float32), np.array([True])

        with (
            patch.object(AdvCPAttackHelper, "load_agent_state", side_effect=lambda scenario_data, agent_id: states[agent_id]),
            patch.object(
                AdvCPAttackHelper,
                "resolve_spoof_boxes_for_agent",
                side_effect=_mock_resolve_spoof_boxes_for_agent,
            ),
            patch.object(
                AdvCoperceptionEarlyFusionAttack,
                "_apply_sampled_ray_traced_spoof",
                side_effect=_mock_apply_sampled_ray_traced_spoof,
            ),
        ):
            result = AdvCoperceptionEarlyFusionAttack.run(
                batch_data,
                model,
                dataset,
                "device(cpu)",
                make_advcp_config(attacker_ids=["cav-2", "cav-3"]),
                memory_data=memory_data,
            )

        first_update = dataset.update_database.call_args_list[0].kwargs["memory_data"]
        restored_update = dataset.update_database.call_args_list[1].kwargs["memory_data"]

        np.testing.assert_array_equal(first_update[0]["cav-2"]["000001"]["lidar_np"], np.array([[9.0, 9.0, 9.0, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(first_update[0]["cav-3"]["000001"]["lidar_np"], np.array([[8.0, 8.0, 8.0, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(first_update[0]["cav-2"]["000001"]["spoofing_mask"], np.array([True]))
        np.testing.assert_array_equal(first_update[0]["cav-3"]["000001"]["spoofing_mask"], np.array([True]))
        assert restored_update is memory_data
        assert batch_data == {"rebuilt": "batch"}
        assert result[3] == {"attacker_ids": ["cav-2", "cav-3"], "fake_box_tensor": None, "mode": "spoof"}
        manager_deps["inference_utils"].inference_early_fusion.assert_called_once_with({"rebuilt": "batch"}, model, dataset)

    def test_intermediate_advcp_run_raises_for_multiple_attackers(self, manager_deps):
        batch_data = {"ego": {"origin_lidar": ["lidar_data"]}}
        dataset = MagicMock()
        model = MagicMock()
        memory_data = {
            0: {
                "cav-1": {"ego": True, "000001": {"params": {"lidar_pose": [0.0] * 6, "true_ego_pos": [0.0] * 6}}},
                "cav-2": {"ego": False, "000001": {"params": {"lidar_pose": [1.0] * 6, "true_ego_pos": [1.0] * 6}}},
                "cav-3": {"ego": False, "000001": {"params": {"lidar_pose": [2.0] * 6, "true_ego_pos": [2.0] * 6}}},
            }
        }

        with pytest.raises(NotImplementedError, match="multiple attackers"):
            AdvCoperceptionIntermediateFusionAttack.run(
                batch_data,
                model,
                dataset,
                "device(cpu)",
                make_advcp_config(attacker_ids=["cav-2", "cav-3"]),
                memory_data=memory_data,
            )

    def test_late_advcp_resolve_spoof_boxes_by_attacker_supports_multiple_attackers(self):
        memory_data = {
            0: {
                "cav-1": {"ego": True, "000001": {"params": {"lidar_pose": [0.0] * 6, "true_ego_pos": [0.0] * 6}}},
                "cav-2": {"ego": False, "000001": {"params": {"lidar_pose": [1.0] * 6, "true_ego_pos": [1.0] * 6}}},
                "cav-3": {"ego": False, "000001": {"params": {"lidar_pose": [2.0] * 6, "true_ego_pos": [2.0] * 6}}},
            }
        }
        spoof_box_2 = np.array([5.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0], dtype=np.float32)
        spoof_box_3 = np.array([8.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0], dtype=np.float32)

        def _mock_resolve_spoof_boxes_for_agent(_scenario_data, _advcp_config, attacker_id):
            spoof_box = spoof_box_2 if attacker_id == "cav-2" else spoof_box_3
            return ("cav-1", MagicMock(), MagicMock(), [spoof_box])

        with patch.object(
            AdvCPAttackHelper,
            "resolve_spoof_boxes_for_agent",
            side_effect=_mock_resolve_spoof_boxes_for_agent,
        ):
            attacker_ids, attack_boxes_by_attacker = AdvCoperceptionLateFusionAttack.resolve_spoof_boxes_by_attacker(
                make_advcp_config(attacker_ids=["cav-2", "cav-3"]),
                memory_data,
            )

        assert attacker_ids == ["cav-2", "cav-3"]
        assert set(attack_boxes_by_attacker.keys()) == {"cav-2", "cav-3"}
        np.testing.assert_array_equal(attack_boxes_by_attacker["cav-2"][0], spoof_box_2)
        np.testing.assert_array_equal(attack_boxes_by_attacker["cav-3"][0], spoof_box_3)

    def test_validate_advcp_agents_supports_multiple_attackers(self, manager_deps):
        opt = DummyOpt(with_advcp=True, advcp_config="dummy.yaml")
        with patch.object(
            AdvCoperceptionModelManager,
            "load_config",
            return_value=make_advcp_config(attacker_ids=["cav-2", "rsu-1"]),
        ):
            manager = AdvCoperceptionModelManager(opt, "2023_01_01")

        assert manager.validate_advcp_agents(["cav-1", "cav-2", "rsu-1"]) is True
        assert manager.advcp_config["attacker_ids"] == ["cav-2", "rsu-1"]

    def test_load_config_rejects_legacy_attacker_id(self, tmp_path):
        config_path = tmp_path / "advcp_legacy.yaml"
        config_path.write_text("mode: spoof\nattacker_id: cav-2\nboxes:\n  - relative: [5, 0, 0, 0, 90, 0]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no longer supported"):
            AdvCoperceptionModelManager.load_config(str(config_path))

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
        assert args[0] == manager.final_result_stat.as_dict()
        assert Path(args[1]).resolve() == expected_dir.resolve()
        assert args[2] == opt.global_sort_detections


class TestCoperceptionVisualizer:
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

    def test_advcp_visualizer_marks_attackers_when_context_is_present(self):
        config = AdvCoperceptionVisualizer.resolve_visualization_config(
            {
                "lidar_point_colors": {
                    "default": (255, 255, 255),
                    "ego": (80, 255, 80),
                    "attackers": (255, 90, 90),
                }
            }
        )
        batch_data = {
            "ego": {
                "origin_lidar_by_agent": [
                    np.array([[1.0, 2.0, 3.0, 1.0]]),
                    np.array([[4.0, 5.0, 6.0, 1.0]]),
                ],
                "origin_lidar_roles": ["ego", "default"],
                "origin_lidar_agent_ids": ["cav-1", "cav-2"],
            }
        }

        _, colors = AdvCoperceptionVisualizer._get_lidar_points_and_colors(
            batch_data,
            None,
            config,
            visualization_context={"attacker_ids": ["cav-2"]},
        )

        assert colors.tolist() == [[80, 255, 80], [255, 90, 90]]

    def test_advcp_visualizer_marks_spoofed_points_with_spoofing_color(self):
        config = AdvCoperceptionVisualizer.resolve_visualization_config(
            {
                "lidar_point_colors": {
                    "default": (255, 255, 255),
                    "ego": (80, 255, 80),
                    "attackers": (255, 90, 90),
                    "spoofing": (242, 156, 74),
                }
            }
        )
        batch_data = {
            "ego": {
                "origin_lidar_by_agent": [
                    np.array([[1.0, 2.0, 3.0, 1.0]]),
                    np.array([[4.0, 5.0, 6.0, 1.0], [7.0, 8.0, 9.0, 1.0]]),
                ],
                "origin_lidar_roles": ["ego", "default"],
                "origin_lidar_agent_ids": ["cav-1", "cav-2"],
                "origin_lidar_spoofing_masks": [
                    np.array([False]),
                    np.array([False, True]),
                ],
            }
        }

        _, colors = AdvCoperceptionVisualizer._get_lidar_points_and_colors(
            batch_data,
            None,
            config,
            visualization_context={"attacker_ids": ["cav-2"]},
        )

        assert colors.tolist() == [[80, 255, 80], [255, 90, 90], [242, 156, 74]]

    def test_advcp_visualizer_uses_other_color_when_special_colors_are_not_configured(self):
        config = AdvCoperceptionVisualizer.resolve_visualization_config(
            {
                "lidar_point_colors": {
                    "other": (255, 70, 0),
                }
            }
        )
        batch_data = {
            "ego": {
                "origin_lidar_by_agent": [
                    np.array([[1.0, 2.0, 3.0, 1.0]]),
                    np.array([[4.0, 5.0, 6.0, 1.0], [7.0, 8.0, 9.0, 1.0]]),
                ],
                "origin_lidar_roles": ["ego", "default"],
                "origin_lidar_agent_ids": ["cav-1", "cav-2"],
                "origin_lidar_spoofing_masks": [
                    np.array([False]),
                    np.array([False, True]),
                ],
            }
        }

        _, colors = AdvCoperceptionVisualizer._get_lidar_points_and_colors(
            batch_data,
            None,
            config,
            visualization_context={"attacker_ids": ["cav-2"]},
        )

        assert colors.tolist() == [[255, 70, 0], [255, 70, 0], [255, 70, 0]]

    def test_early_advcp_density_aliases_are_supported(self):
        assert AdvCoperceptionEarlyFusionAttack._resolve_density(0) == 0
        assert AdvCoperceptionEarlyFusionAttack._resolve_density("dense_a") == 1
        assert AdvCoperceptionEarlyFusionAttack._resolve_density("dense_all") == 2
        assert AdvCoperceptionEarlyFusionAttack._resolve_density("sampled") == 3

    def test_early_advcp_density_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            AdvCoperceptionEarlyFusionAttack._resolve_density("mystery")

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
        localization_manager = MagicMock(spec_set=["get_ego_pos", "get_ego_spd", "vehicle"])
        localization_manager.get_ego_pos.return_value = predicted_ego_pos
        localization_manager.get_ego_spd.return_value = 13.5
        localization_manager.vehicle = MagicMock(get_transform=MagicMock(return_value=true_ego_pos))

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
            params = processor.build_live_params(perception_manager, localization_manager, behavior_agent)

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
        localization_manager = MagicMock(spec_set=["get_ego_pos", "get_ego_spd", "rsu", "true_ego_pos"])
        localization_manager.get_ego_pos.return_value = predicted_ego_pos
        localization_manager.get_ego_spd.return_value = 0.0
        localization_manager.rsu = MagicMock()
        localization_manager.true_ego_pos = true_ego_pos

        params = processor.build_live_params(perception_manager, localization_manager, None)

        assert params["RSU"] is True
        assert "plan_trajectory" not in params
        assert params["true_ego_pos"] == (51.0, 61.0, 71.0, 0.0, 91.0, 0.0)

    def test_build_live_params_raises_for_unknown_localizer_type(self):
        processor = CoperceptionDataProcessor()
        perception_manager = MagicMock(
            objects={"vehicles": []}, lidar=MagicMock(sensor=MagicMock(get_transform=MagicMock(return_value=self._make_transform())))
        )
        localization_manager = MagicMock(spec_set=["get_ego_pos", "get_ego_spd"])
        localization_manager.get_ego_pos.return_value = self._make_transform()
        localization_manager.get_ego_spd.return_value = 0.0

        with pytest.raises(ValueError, match="Unknown localization manager type"):
            processor.build_live_params(perception_manager, localization_manager, None)

    def test_build_live_memory_returns_none_and_warns_for_empty_agents(self):
        processor = CoperceptionDataProcessor()

        with patch("opencda.core.common.coperception_data_processor.logger.warning") as mock_warning:
            memory = processor.build_live_memory([], [], 5)

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
        cav1.perception_manager = MagicMock(lidar=MagicMock(data=np.array([[1.0, 2.0, 3.0, 1.0]])))
        cav1.localizer = MagicMock()
        cav1.agent = MagicMock()

        cav2 = MagicMock()
        cav2.id = "cav-2"
        cav2.perception_manager = MagicMock(lidar=MagicMock(data=np.array([[4.0, 5.0, 6.0, 1.0]])))
        cav2.localizer = MagicMock()
        cav2.agent = MagicMock()

        rsu = MagicMock()
        rsu.id = "rsu-1"
        rsu.perception_manager = MagicMock(lidar=MagicMock(data=np.array([[7.0, 8.0, 9.0, 1.0]])))
        rsu.localizer = MagicMock()

        with (
            patch.object(CoperceptionDataProcessor, "build_live_params", side_effect=[{"id": 1}, {"id": 2}, {"id": 3}]),
            patch.object(
                CoperceptionDataProcessor,
                "_build_live_camera_snapshots",
                return_value=[],
            ),
        ):
            memory = processor.build_live_memory([cav1, cav2], [rsu], 12)

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
        cav.perception_manager = MagicMock(lidar=None)
        cav.localizer = MagicMock()
        cav.agent = MagicMock()

        with patch("opencda.core.common.coperception_data_processor.logger.warning") as mock_warning:
            memory = processor.build_live_memory([cav], [], 1)

        assert memory is None
        assert mock_warning.call_count == 2
        assert "LiDAR is not initialized" in mock_warning.call_args_list[0].args[0]
        assert "no agents have valid LiDAR data" in mock_warning.call_args_list[1].args[0]

    def test_build_live_memory_skips_agent_when_vehicle_lidar_data_is_missing(self):
        processor = CoperceptionDataProcessor()
        cav = MagicMock()
        cav.id = "cav-1"
        cav.perception_manager = MagicMock(lidar=MagicMock(data=None))
        cav.localizer = MagicMock()
        cav.agent = MagicMock()

        with patch("opencda.core.common.coperception_data_processor.logger.warning") as mock_warning:
            memory = processor.build_live_memory([cav], [], 1)

        assert memory is None
        assert mock_warning.call_count == 2
        assert "LiDAR data is not initialized" in mock_warning.call_args_list[0].args[0]
        assert "no agents have valid LiDAR data" in mock_warning.call_args_list[1].args[0]
