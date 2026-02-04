"""
Cooperative perception model manager for multi-agent object detection.

This module provides functionality for running cooperative perception models
using OpenCOOD framework, including dataset building, inference, evaluation,
and visualization capabilities.
"""

import os
import re
import shutil
import logging
from typing import Dict, List, Optional, Any
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
    """
    Cooperative perception model manager for multi-agent object detection.

    Manages the entire pipeline for cooperative perception including model loading,
    dataset creation, inference, evaluation, and visualization.

    Parameters
    ----------
    opt : Any
        Configuration options containing model paths and fusion methods.
    current_time : str
        Timestamp string for organizing output files.
    message_handler : Any, optional
        Handler for V2X message processing. Default is None.

    Attributes
    ----------
    opt : Any
        Configuration options.
    hypes : Dict[str, Any]
        Hyperparameters loaded from YAML configuration.
    model : torch.nn.Module
        Cooperative perception neural network model.
    current_time : str
        Timestamp for output organization.
    vis : o3d.visualization.Visualizer or None
        Open3D visualizer for sequence visualization.
    device : torch.device
        Computing device (CPU or CUDA).
    saved_path : str
        Path to saved model checkpoint.
    opencood_dataset : Any or None
        OpenCOOD dataset instance.
    data_loader : DataLoader or None
        PyTorch DataLoader for batch processing.
    message_handler : Any or None
        V2X message handler.
    final_result_stat : [float, Dict[str, Union[List[Any], int]]]
        Accumulated evaluation statistics across all predictions.
    """

    def __init__(self, opt: Any, current_time: str, message_handler: Optional[Any] = None):
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

        self.final_result_stat: Dict[float, Dict[str, Any]] = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

    def make_dataset(self) -> None:
        """
        Build the OpenCOOD dataset and create DataLoader.

        Creates dataset instance from configuration and initializes
        DataLoader for batch processing.
        """
        logger.info("Dataset Building")
        self.opencood_dataset = build_dataset(self.hypes, visualize=True, train=False, message_handler=self.message_handler)
        logger.info(f"{len(self.opencood_dataset)} samples found.")  # NOTE None-check is required
        self.data_loader = DataLoader(
            self.opencood_dataset,
            batch_size=1,
            num_workers=16,
            collate_fn=self.opencood_dataset.collate_batch_test,  # NOTE None-check is required
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def make_prediction(self, tick_number: int) -> None:
        """
        Run cooperative perception inference on the dataset.

        Performs model inference using specified fusion method, evaluates predictions,
        and optionally saves/visualizes results.

        Parameters
        ----------
        tick_number : int
            Current simulation tick number for naming output files.

        Raises
        ------
        AssertionError
            If fusion method is not one of 'late', 'early', 'intermediate'.
        AssertionError
            If both show_vis and show_sequence options are enabled.
        NotImplementedError
            If fusion method is not supported.
        """
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

        if self.opt.show_sequence:  # NOTE None-check is required
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

        for i, batch_data in tqdm(
            enumerate(self.data_loader), total=len(self.data_loader)
        ):  # NOTE Argument 1 to "len" has incompatible type "None"; expected "Sized
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.device)
                if self.opt.fusion_method == "late":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(batch_data, self.model, self.opencood_dataset)
                elif self.opt.fusion_method == "early":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data, self.model, self.opencood_dataset)
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
                    self.opencood_dataset.visualize_result(  # NOTE None-check is required
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data["ego"]["origin_lidar"],
                        self.opt.show_vis,
                        vis_save_path,
                        dataset=self.opencood_dataset,
                    )

                if self.opt.show_sequence and pred_box_tensor is not None and self.hypes["postprocess"]["core_method"] != "BevPostprocessor":
                    self.vis.clear_geometries()  # NOTE None-check is required
                    pcd, pred_o3d_box, gt_o3d_box = vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"], vis_pcd, mode="constant"
                    )
                    if i == 0:
                        self.vis.add_geometry(pcd)  # NOTE None-check is required
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_pred, pred_o3d_box, update_mode="add")
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_gt, gt_o3d_box, update_mode="add")
                    else:
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_pred, pred_o3d_box)
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_gt, gt_o3d_box)
                    self.vis.update_geometry(pcd)  # NOTE None-check is required
                    self.vis.poll_events()  # NOTE None-check is required
                    self.vis.update_renderer()  # NOTE None-check is required

        for iou in [0.3, 0.5, 0.7]:
            self.final_result_stat[iou]["gt"] += result_stat[iou]["gt"]
            self.final_result_stat[iou]["tp"] += result_stat[iou]["tp"]
            self.final_result_stat[iou]["fp"] += result_stat[iou]["fp"]
            self.final_result_stat[iou]["score"] += result_stat[iou]["score"]

    def final_eval(self) -> None:
        """
        Perform final evaluation and save results.

        Evaluates accumulated predictions across all batches and saves
        evaluation metrics to disk.
        """
        eval_dir = f"simulation_output/coperception/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)
        eval_utils.eval_final_results(self.final_result_stat, eval_dir, self.opt.global_sort_detections)


class DirectoryProcessor:
    """
    Directory processor for managing simulation data dumps.

    Handles copying and organizing sensor data files from data dumping directory
    to processing directory for cooperative perception inference.

    Parameters
    ----------
    source_directory : str, optional
        Source directory containing dumped simulation data. Default is "data_dumping".
    now_directory : str, optional
        Target directory for current processing data. Default is "data_dumping/sample/now".

    Attributes
    ----------
    source_directory : str
        Path to source data directory.
    now_directory : str
        Path to current processing directory.
    """

    def __init__(
        self,
        source_directory: str = "data_dumping",
        now_directory: str = "data_dumping/sample/now",
    ):
        self.source_directory = source_directory
        self.now_directory = now_directory

    def detect_cameras(self, data_directory: str) -> List[str]:
        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])
        if not inner_subdirectories:
            return []

        sample_folder = os.path.join(data_directory, inner_subdirectories[0])
        camera_files = [f for f in os.listdir(sample_folder) if re.match(r"\d+_camera\d+\.png", f)]

        camera_ids = sorted(set(re.findall(r"_camera(\d+)\.png", f)[0] for f in camera_files if re.findall(r"_camera(\d+)\.png", f)))

        return [f"_camera{cam_id}.png" for cam_id in camera_ids]

    def process_directory(self, tick_number: int) -> None:
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

    def clear_directory_now(self) -> None:
        for item in os.listdir(self.now_directory):
            item_path = os.path.join(self.now_directory, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Удаляем файлы и символические ссылки
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
