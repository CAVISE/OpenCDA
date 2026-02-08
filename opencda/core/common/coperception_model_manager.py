import os
import re
import shutil
import logging
from typing import Dict, Optional
from tqdm import tqdm

import torch  # type: ignore
import open3d as o3d
from torch.utils.data import DataLoader  # type: ignore

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis, vis_utils
from opencood.utils import eval_utils

logger = logging.getLogger("cavise.coperception_model_manager")


class CoperceptionModelManager:
    def __init__(self, opt, current_time, message_handler=None):
        self.opt = opt
        self.hypes = yaml_utils.load_yaml(None, self.opt)
        self.model = train_utils.create_model(self.hypes)
        self.current_time = current_time
        self.vis = None

        if torch.cuda.is_available():
            self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saved_path = self.opt.model_dir
        _, self.model = train_utils.load_saved_model(self.saved_path, self.model)

        self.opencood_dataset = None
        self.data_loader = None
        self.message_handler = message_handler

        # Store current batch data to avoid circular dependency with AdvCP
        self._current_batch_data = None
        self._current_batch_index = None

        logger.info("Initial Dataset Building")
        self.opencood_dataset = build_dataset(self.hypes, visualize=True, train=False, message_handler=self.message_handler)

        self.data_loader = DataLoader(
            self.opencood_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=self.opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        self.final_result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        # Initialize AdvCP Manager if enabled
        self.advcp_manager = None
        if self.opt.get("with_advcp", False):
            from opencda.core.common.advcp.advcp_manager import AdvCPManager
            from opencda.core.common.advcp.advcp_config import load_advcp_config

            advcp_config = load_advcp_config(self.opt)
            self.advcp_manager = AdvCPManager(advcp_config, current_time, self, message_handler)
            logger.info("AdvCP Manager initialized and integrated with CoperceptionManager")

    def update_dataset(self):
        logger.debug("Refreshing dataset indices")
        self.opencood_dataset.update_database()

        if len(self.opencood_dataset) == 0:
            logger.warning("No samples found in dataset after update.")

    def _get_raw_data(self, tick_number: int) -> Optional[Dict]:
        """
        Get raw perception data for a specific tick without running prediction.
        This method is used by AdvCPManager to avoid circular dependency.

        Args:
            tick_number: The tick number to retrieve data for

        Returns:
            Dictionary containing raw perception data, or None if data not available
        """
        # Check if we have stored data for the requested tick
        if self._current_batch_index == tick_number and self._current_batch_data is not None:
            return self._current_batch_data

        # If not, try to get it from the dataset directly
        try:
            # Access the dataset directly to get raw data
            if tick_number < len(self.opencood_dataset):
                raw_data = self.opencood_dataset[tick_number]
                return raw_data
        except Exception as e:
            logger.warning(f"Failed to get raw data for tick {tick_number}: {e}")

        return None

    def make_prediction(self, tick_number):
        assert self.opt.fusion_method in ["late", "early", "intermediate"]
        assert not (self.opt.show_vis and self.opt.show_sequence), "you can only visualize the results in single image mode or video mode"
        self.model.eval()

        # Create the dictionary for evaluation.
        # also store the confidence score for each prediction
        result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        if self.opt.show_sequence:
            if self.vis is None:
                self.vis = o3d.visualization.Visualizer()  # noqa: DC05
                self.vis.create_window()  # noqa: DC05
                self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]  # noqa: DC05
                self.vis.get_render_option().point_size = 1.0  # noqa: DC05
                self.vis.get_render_option().show_coordinate_frame = True  # noqa: DC05
            # used to visualize lidar points
            vis_pcd = o3d.geometry.PointCloud()
            # used to visualize object bounding box, maximum 50
            vis_aabbs_gt = []
            vis_aabbs_pred = []
            for _ in range(50):
                vis_aabbs_gt.append(o3d.geometry.LineSet())
                vis_aabbs_pred.append(o3d.geometry.LineSet())

        for i, batch_data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            with torch.no_grad():
                # Store current batch data for AdvCPManager to avoid circular dependency
                self._current_batch_index = i
                self._current_batch_data = batch_data

                batch_data = train_utils.to_device(batch_data, self.device)

                # Apply AdvCP if enabled
                if self.advcp_manager and self.advcp_manager.with_advcp:
                    # Get original prediction
                    if self.opt.fusion_method == "late":
                        original_pred_box_tensor, original_pred_score, original_gt_box_tensor = inference_utils.inference_late_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                    elif self.opt.fusion_method == "early":
                        original_pred_box_tensor, original_pred_score, original_gt_box_tensor = inference_utils.inference_early_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                    elif self.opt.fusion_method == "intermediate":
                        original_pred_box_tensor, original_pred_score, original_gt_box_tensor = inference_utils.inference_intermediate_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                    else:
                        raise NotImplementedError("Only early, late and intermediate fusion is supported.")

                    # Apply AdvCP attacks/defenses
                    # Prepare predictions for late attacks
                    predictions = None
                    if self.advcp_manager.attack_type in ["lidar_remove_late", "lidar_spoof_late"]:
                        predictions = {
                            "pred_bboxes": original_pred_box_tensor,
                            "pred_scores": original_pred_score,
                            "gt_bboxes": original_gt_box_tensor
                        }
                    
                    modified_data, defense_score, defense_metrics = self.advcp_manager.process_tick(
                        i, 
                        batch_data=batch_data,
                        predictions=predictions
                    )

                    if modified_data:
                        # Extract modified predictions from AdvCP data
                        pred_box_tensor = None
                        pred_score = None
                        gt_box_tensor = None

                        for vehicle_id, vehicle_data in modified_data.items():
                            if "pred_bboxes" in vehicle_data and "pred_scores" in vehicle_data:
                                if pred_box_tensor is None:
                                    pred_box_tensor = torch.from_numpy(vehicle_data["pred_bboxes"]).to(self.device)
                                    pred_score = torch.from_numpy(vehicle_data["pred_scores"]).to(self.device)
                                else:
                                    pred_box_tensor = torch.cat(
                                        [pred_box_tensor, torch.from_numpy(vehicle_data["pred_bboxes"]).to(self.device)], dim=0
                                    )
                                    pred_score = torch.cat([pred_score, torch.from_numpy(vehicle_data["pred_scores"]).to(self.device)], dim=0)

                        if pred_box_tensor is None:
                            # Fallback to original predictions if AdvCP failed
                            pred_box_tensor = original_pred_box_tensor
                            pred_score = original_pred_score
                            gt_box_tensor = original_gt_box_tensor
                        else:
                            # Use modified predictions from AdvCP
                            gt_box_tensor = original_gt_box_tensor
                    else:
                        # No AdvCP applied, use original predictions
                        pred_box_tensor = original_pred_box_tensor
                        pred_score = original_pred_score
                        gt_box_tensor = original_gt_box_tensor
                else:
                    # No AdvCP, use original predictions
                    if self.opt.fusion_method == "late":
                        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                    elif self.opt.fusion_method == "early":
                        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                    elif self.opt.fusion_method == "intermediate":
                        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                    else:
                        raise NotImplementedError("Only early, late and intermediate fusion is supported.")

                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

                if self.opt.save_npy:
                    npy_dir = f"simulation_output/coperception/npy/{self.opt.test_scenario}_{self.current_time}"
                    npy_save_path = os.path.join(npy_dir, "npy")
                    os.makedirs(npy_save_path, exist_ok=True)
                    inference_utils.save_prediction_gt(pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"][0], i, npy_save_path)

                if self.opt.save_vis:
                    for mode in ["3d", "bev"]:
                        if self.hypes["postprocess"]["core_method"] == "BevPostprocessor" and mode == "3d":
                            continue
                        vis_dir = f"simulation_output/coperception/vis_{mode}/{self.opt.test_scenario}_{self.current_time}"
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_save_path = os.path.join(vis_dir, f"{mode}_{tick_number:05d}.png")
                        simple_vis.visualize(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"][0],
                            self.hypes["postprocess"]["gt_range"],
                            vis_save_path,
                            method=mode,
                            left_hand=True,
                            vis_pred_box=True,
                        )

                if self.opt.show_vis:
                    vis_save_path = ""
                    self.opencood_dataset.visualize_result(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data["ego"]["origin_lidar"],
                        self.opt.show_vis,
                        vis_save_path,
                        dataset=self.opencood_dataset,
                    )

                if self.opt.show_sequence and pred_box_tensor is not None and self.hypes["postprocess"]["core_method"] != "BevPostprocessor":
                    self.vis.clear_geometries()
                    pcd, pred_o3d_box, gt_o3d_box = vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"], vis_pcd, mode="constant"
                    )
                    if i == 0:
                        self.vis.add_geometry(pcd)
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_pred, pred_o3d_box, update_mode="add")
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_gt, gt_o3d_box, update_mode="add")
                    else:
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_pred, pred_o3d_box)
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_gt, gt_o3d_box)
                    self.vis.update_geometry(pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()

        for iou in [0.3, 0.5, 0.7]:
            self.final_result_stat[iou]["gt"] += result_stat[iou]["gt"]
            self.final_result_stat[iou]["tp"] += result_stat[iou]["tp"]
            self.final_result_stat[iou]["fp"] += result_stat[iou]["fp"]
            self.final_result_stat[iou]["score"] += result_stat[iou]["score"]

        # Update AdvCP statistics if enabled
        if self.advcp_manager and self.advcp_manager.with_advcp:
            attack_stats = self.advcp_manager.get_attack_statistics()
            defense_stats = self.advcp_manager.get_defense_statistics()

            if attack_stats:
                logger.info(f"AdvCP Attack Statistics: {attack_stats}")
            if defense_stats:
                logger.info(f"AdvCP Defense Statistics: {defense_stats}")

    def final_eval(self):
        eval_dir = f"simulation_output/coperception/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)
        eval_utils.eval_final_results(self.final_result_stat, eval_dir, self.opt.global_sort_detections)


class DirectoryProcessor:
    def __init__(self, source_directory="data_dumping", now_directory="data_dumping/sample/now"):
        self.source_directory = source_directory
        self.now_directory = now_directory

    def detect_cameras(self, data_directory):
        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])
        if not inner_subdirectories:
            return []

        sample_folder = os.path.join(data_directory, inner_subdirectories[0])
        camera_files = [f for f in os.listdir(sample_folder) if re.match(r"\d+_camera\d+\.png", f)]

        camera_ids = sorted(set(re.findall(r"_camera(\d+)\.png", f)[0] for f in camera_files if re.findall(r"_camera(\d+)\.png", f)))

        return [f"_camera{cam_id}.png" for cam_id in camera_ids]

    def process_directory(self, tick_number):
        number = f"{tick_number:06d}"
        postfixes = [".pcd", ".yaml"]

        subdirectories = sorted([d for d in os.listdir(self.source_directory) if os.path.isdir(os.path.join(self.source_directory, d))])

        if len(subdirectories) < 2:
            raise ValueError("Not enough subdirectories in source directory to process.")

        data_directory = os.path.join(self.source_directory, subdirectories[-2])

        camera_postfixes = self.detect_cameras(data_directory)
        postfixes.extend(camera_postfixes)

        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])

        shutil.copy(os.path.join(data_directory, "data_protocol.yaml"), self.now_directory)

        for folder in inner_subdirectories:
            destination_folder = os.path.join(self.now_directory, folder)
            os.makedirs(destination_folder, exist_ok=True)
            for postfix in postfixes:
                source_file_path = os.path.join(data_directory, folder, f"{number}{postfix}")
                destination_file_path = os.path.join(destination_folder, f"{number}{postfix}")
                if os.path.exists(source_file_path):
                    shutil.copy(source_file_path, destination_file_path)

    def clear_directory_now(self):
        for item in os.listdir(self.now_directory):
            item_path = os.path.join(self.now_directory, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Удаляем файлы и символические ссылки
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
