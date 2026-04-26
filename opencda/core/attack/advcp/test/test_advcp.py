import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import opencda.core.common.coperception_model_manager as coperception_model_manager_module
import opencda.core.attack.advcp.early_fusion_attack as early_fusion_attack_module
import opencda.core.attack.advcp.intermediate_fusion_attack as intermediate_fusion_attack_module

# The production code imports are now safe because pytest_configure in conftest.py
# installs the mocks before collection.
from opencda.core.attack.advcp import AdvCoperceptionModelManager
from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.adv_coperception_model_manager import AdvCoperceptionVisualizer
from opencda.core.attack.advcp.early_fusion_attack import AdvCoperceptionEarlyFusionAttack
from opencda.core.attack.advcp.intermediate_fusion_attack import AdvCoperceptionIntermediateFusionAttack
from opencda.core.attack.advcp.late_fusion_attack import AdvCoperceptionLateFusionAttack
from opencda.core.common.coperception_model_manager import CoperceptionInferenceResult


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


@pytest.fixture
def manager_deps(fake_heavy_deps):
    """
    Setup mocks specifically for Manager instantiation and method calls.
    Resets mocks before every test to ensure isolation.
    """
    opencood = fake_heavy_deps["opencood"]
    torch = fake_heavy_deps["torch"]
    open3d = fake_heavy_deps["open3d"]

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

    for m in mocks_to_reset:
        m.reset_mock()

    opencood.utils.eval_utils.caluclate_tp_fp.side_effect = None

    hypes = {
        "postprocess": {"core_method": "VoxelPostprocessor", "gt_range": [0, -40, -3, 70, 40, 1]},
        "fusion": {"core_method": "LateFusionDataset"},
    }
    opencood.hypes_yaml.yaml_utils.load_yaml.return_value = hypes

    model = MagicMock()
    opencood.tools.train_utils.create_model.return_value = model
    opencood.tools.train_utils.load_saved_model.return_value = (None, model)

    # Re-bind module-level imports so assertions in this test file always
    # observe the same mock objects even during full-suite execution.
    coperception_model_manager_module.torch = torch
    coperception_model_manager_module.o3d = open3d
    coperception_model_manager_module.yaml_utils = opencood.hypes_yaml.yaml_utils
    coperception_model_manager_module.train_utils = opencood.tools.train_utils
    coperception_model_manager_module.inference_utils = opencood.tools.inference_utils
    coperception_model_manager_module.build_dataset = opencood.data_utils.datasets.build_dataset
    coperception_model_manager_module.vis_utils = opencood.visualization.vis_utils
    coperception_model_manager_module.eval_utils = opencood.utils.eval_utils
    early_fusion_attack_module.inference_utils = opencood.tools.inference_utils
    early_fusion_attack_module.train_utils = opencood.tools.train_utils
    intermediate_fusion_attack_module.inference_utils = opencood.tools.inference_utils
    intermediate_fusion_attack_module.train_utils = opencood.tools.train_utils

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


class TestAdvCoperceptionModelManager:
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


class TestAdvCoperceptionVisualizer:
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


class TestAdvCoperceptionEarlyFusionAttack:
    def test_early_advcp_density_aliases_are_supported(self):
        assert AdvCoperceptionEarlyFusionAttack._resolve_density(0) == 0
        assert AdvCoperceptionEarlyFusionAttack._resolve_density("dense_a") == 1
        assert AdvCoperceptionEarlyFusionAttack._resolve_density("dense_all") == 2
        assert AdvCoperceptionEarlyFusionAttack._resolve_density("sampled") == 3

    def test_early_advcp_density_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            AdvCoperceptionEarlyFusionAttack._resolve_density("mystery")
