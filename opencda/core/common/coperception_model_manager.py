from __future__ import annotations

import copy
import os
import re
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Tuple, TypeAlias, cast
from tqdm import tqdm
from collections import OrderedDict

from matplotlib import pyplot as plt
import numpy as np
import torch  # type: ignore
import open3d as o3d
from torch.utils.data import DataLoader  # type: ignore

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from opencda.core.attack.advcp import (
    inference_early_fusion_attack,
    inference_intermediate_fusion_attack,
    inference_late_fusion_attack,
    load_advcp_config,
)
from opencda.core.attack.advcp.utils import resolve_attacker_id

if TYPE_CHECKING:
    from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset

logger = logging.getLogger("cavise.opencda.opencda.core.common.coperception_model_manager")


class CoperceptionVisualizer:
    _DEFAULT_VISUALIZATION_CONFIG: Dict[str, Any] = {
        "background": [0, 0, 0],
        "lidar_point_colors": {
            "default": [255, 255, 255],
            "ego": [80, 255, 80],
            "attackers": [255, 90, 90],
            "spoofing": [180, 0, 255],
        },
        "bbox_colors": {
            "gt": [0, 255, 0],
            "pred": [255, 0, 0],
            "fake": [180, 0, 255],
        },
    }

    @staticmethod
    def resolve_visualization_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        resolved = copy.deepcopy(CoperceptionVisualizer._DEFAULT_VISUALIZATION_CONFIG)
        if not config:
            return resolved

        config_dict = dict(config)
        for key in ("background",):
            if key in config_dict and config_dict[key] is not None:
                resolved[key] = list(config_dict[key])

        for key in ("lidar_point_colors", "bbox_colors"):
            value = config_dict.get(key)
            if isinstance(value, Mapping):
                resolved[key].update(dict(value))

        return resolved

    @staticmethod
    def render_inference_to_file(
        pred_box_tensor,
        gt_tensor,
        pcd,
        pc_range,
        save_path,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
        method: str = "3d",
        vis_gt_box: bool = True,
        vis_pred_box: bool = True,
        left_hand: bool = False,
        uncertainty=None,
    ):
        config = CoperceptionVisualizer.resolve_visualization_config(visualization_config)
        canvas = CoperceptionVisualizer._build_canvas(
            pred_box_tensor=pred_box_tensor,
            gt_tensor=gt_tensor,
            pcd=pcd,
            pc_range=pc_range,
            batch_data=batch_data,
            visualization_config=config,
            method=method,
            vis_gt_box=vis_gt_box,
            vis_pred_box=vis_pred_box,
            left_hand=left_hand,
            uncertainty=uncertainty,
        )
        plt.axis("off")
        plt.imshow(canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=400, pad_inches=0.0)
        plt.clf()

    @staticmethod
    def show_inference_image(
        pred_box_tensor,
        gt_tensor,
        pcd,
        pc_range,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
        method: str = "3d",
        vis_gt_box: bool = True,
        vis_pred_box: bool = True,
        left_hand: bool = False,
        uncertainty=None,
    ):
        config = CoperceptionVisualizer.resolve_visualization_config(visualization_config)
        canvas = CoperceptionVisualizer._build_canvas(
            pred_box_tensor=pred_box_tensor,
            gt_tensor=gt_tensor,
            pcd=pcd,
            pc_range=pc_range,
            batch_data=batch_data,
            visualization_config=config,
            method=method,
            vis_gt_box=vis_gt_box,
            vis_pred_box=vis_pred_box,
            left_hand=left_hand,
            uncertainty=uncertainty,
        )
        plt.axis("off")
        plt.imshow(canvas)
        plt.tight_layout()
        plt.show()
        plt.clf()

    @staticmethod
    def visualize_inference_sample_dataloader(
        pred_box_tensor,
        gt_box_tensor,
        origin_lidar,
        o3d_pcd,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
    ):
        config = CoperceptionVisualizer.resolve_visualization_config(visualization_config)
        origin_lidar_np, point_colors_uint8 = CoperceptionVisualizer._get_lidar_points_and_colors(batch_data, origin_lidar, config)
        pred_color = CoperceptionVisualizer._as_float_color(config["bbox_colors"]["pred"])
        gt_color = CoperceptionVisualizer._as_float_color(config["bbox_colors"]["gt"])

        origin_lidar_np = np.array(origin_lidar_np, copy=True)
        origin_lidar_np[:, :1] = -origin_lidar_np[:, :1]
        point_colors = point_colors_uint8.astype(np.float64) / 255.0
        o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar_np[:, :3])
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_colors)

        pred_o3d_box = vis_utils.bbx2linset(pred_box_tensor, color=pred_color) if pred_box_tensor is not None else []
        gt_o3d_box = vis_utils.bbx2linset(gt_box_tensor, color=gt_color) if gt_box_tensor is not None else []

        return o3d_pcd, pred_o3d_box, gt_o3d_box

    @staticmethod
    def _build_canvas(
        pred_box_tensor,
        gt_tensor,
        pcd,
        pc_range,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
        method: str = "3d",
        vis_gt_box: bool = True,
        vis_pred_box: bool = True,
        left_hand: bool = False,
        uncertainty=None,
    ):
        import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
        import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
        from opencood.utils import common_utils

        config = CoperceptionVisualizer.resolve_visualization_config(visualization_config)
        pc_range = [int(i) for i in pc_range]
        pcd_np, point_colors = CoperceptionVisualizer._get_lidar_points_and_colors(batch_data, pcd, config)
        bg_color = CoperceptionVisualizer._as_uint8_color(config["background"])
        gt_color = CoperceptionVisualizer._as_uint8_color(config["bbox_colors"]["gt"])
        pred_color = CoperceptionVisualizer._as_uint8_color(config["bbox_colors"]["pred"])

        if vis_pred_box and pred_box_tensor is not None:
            pred_box_np = common_utils.torch_tensor_to_numpy(pred_box_tensor)
            pred_name = [""] * pred_box_np.shape[0]
            if uncertainty is not None:
                uncertainty_np = common_utils.torch_tensor_to_numpy(uncertainty)
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2

                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:, :2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np)
                    pred_name = [
                        f"x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}"
                        for i in range(uncertainty_np.shape[0])
                    ]
                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:, :2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np)
                    pred_name = [f"x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}" for i in range(uncertainty_np.shape[0])]
                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:, :2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np)
                    pred_name = [
                        f"x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}"
                        for i in range(uncertainty_np.shape[0])
                    ]
        else:
            pred_box_np = None
            pred_name = []

        if vis_gt_box:
            gt_box_np = common_utils.torch_tensor_to_numpy(gt_tensor)
            gt_name = [""] * gt_box_np.shape[0]

        if method == "bev":
            canvas = canvas_bev.Canvas_BEV_heading_right(
                canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
                canvas_x_range=(pc_range[0], pc_range[3]),
                canvas_y_range=(pc_range[1], pc_range[4]),
                canvas_bg_color=bg_color,
                left_hand=left_hand,
            )
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            if valid_mask.any():
                canvas.draw_canvas_points(canvas_xy[valid_mask], colors=point_colors[valid_mask])
            if vis_gt_box and gt_box_np is not None:
                canvas.draw_boxes(gt_box_np, colors=gt_color, texts=gt_name, box_line_thickness=5)
            if vis_pred_box and pred_box_np is not None:
                canvas.draw_boxes(pred_box_np, colors=pred_color, texts=pred_name, box_line_thickness=5)
        elif method == "3d":
            canvas = canvas_3d.Canvas_3D(canvas_bg_color=bg_color, left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            if valid_mask.any():
                canvas.draw_canvas_points(canvas_xy[valid_mask], colors=point_colors[valid_mask])
            if vis_gt_box and gt_box_np is not None:
                canvas.draw_boxes(gt_box_np, colors=gt_color, texts=gt_name)
            if vis_pred_box and pred_box_np is not None:
                canvas.draw_boxes(pred_box_np, colors=pred_color, texts=pred_name)
        else:
            raise ValueError(f"Unsupported visualization method: {method}")

        return canvas.canvas

    @staticmethod
    def _to_numpy_points(pcd) -> np.ndarray:
        if isinstance(pcd, list):
            from opencood.utils import common_utils

            pcd_np = [common_utils.torch_tensor_to_numpy(x) for x in pcd]
            pcd_np = pcd_np[0]
        else:
            if isinstance(pcd, np.ndarray):
                pcd_np = pcd
            else:
                from opencood.utils import common_utils

                pcd_np = common_utils.torch_tensor_to_numpy(pcd)

        if len(pcd_np.shape) > 2:
            pcd_np = pcd_np[0]
        if pcd_np.shape[1] > 4:
            pcd_np = pcd_np[:, 1:]
        return np.array(pcd_np, copy=True)

    @staticmethod
    def _get_lidar_points_and_colors(batch_data, fallback_pcd, config: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        default_color = CoperceptionVisualizer._as_uint8_color(config["lidar_point_colors"]["default"])
        ego_color = CoperceptionVisualizer._as_uint8_color(config["lidar_point_colors"].get("ego", default_color))

        if isinstance(batch_data, Mapping):
            ego_entry = batch_data.get("ego")
            if isinstance(ego_entry, Mapping) and "origin_lidar_by_agent" in ego_entry:
                points_by_agent = ego_entry["origin_lidar_by_agent"]
                roles = list(ego_entry.get("origin_lidar_roles", []))
                agent_ids = list(ego_entry.get("origin_lidar_agent_ids", []))
                colored_points = []
                colored_values = []

                for idx, points in enumerate(points_by_agent):
                    agent_points = CoperceptionVisualizer._to_numpy_points(points)
                    if agent_points.size == 0:
                        continue

                    role = roles[idx] if idx < len(roles) else "default"
                    agent_id = agent_ids[idx] if idx < len(agent_ids) else None
                    color = CoperceptionVisualizer._resolve_point_color(
                        config, agent_id=agent_id, role=role, default_color=default_color, ego_color=ego_color
                    )
                    colored_points.append(agent_points)
                    colored_values.append(np.tile(np.asarray(color, dtype=np.uint8), (agent_points.shape[0], 1)))

                if colored_points:
                    return np.vstack(colored_points), np.vstack(colored_values)

            colored_points = []
            colored_values = []
            for cav_id, cav_content in batch_data.items():
                if not isinstance(cav_content, Mapping):
                    continue
                local_lidar = cav_content.get("origin_lidar_local")
                if local_lidar is None:
                    continue

                cav_points = CoperceptionVisualizer._to_numpy_points(local_lidar)
                if cav_points.size == 0:
                    continue

                if cav_id != "ego":
                    from opencood.utils import box_utils

                    transformation_matrix = cav_content.get("transformation_matrix")
                    if transformation_matrix is not None:
                        transformation_matrix_np = CoperceptionVisualizer._to_numpy_array(transformation_matrix)
                        cav_points[:, :3] = box_utils.project_points_by_matrix_torch(cav_points[:, :3], transformation_matrix_np)

                cav_color = CoperceptionVisualizer._resolve_point_color(
                    config, agent_id=cav_id, role="ego" if cav_id == "ego" else "default", default_color=default_color, ego_color=ego_color
                )
                colored_points.append(cav_points)
                colored_values.append(np.tile(np.asarray(cav_color, dtype=np.uint8), (cav_points.shape[0], 1)))

            if colored_points:
                return np.vstack(colored_points), np.vstack(colored_values)

        fallback_points = CoperceptionVisualizer._to_numpy_points(fallback_pcd)
        fallback_colors = np.tile(np.asarray(default_color, dtype=np.uint8), (fallback_points.shape[0], 1))
        return fallback_points, fallback_colors

    @staticmethod
    def _as_uint8_color(color: Iterable[Any]) -> Tuple[int, int, int]:
        rgb = [int(v) for v in list(color)[:3]]
        return tuple(max(0, min(255, value)) for value in rgb)

    @staticmethod
    def _as_float_color(color: Iterable[Any]) -> Tuple[float, float, float]:
        return tuple(channel / 255.0 for channel in CoperceptionVisualizer._as_uint8_color(color))

    @staticmethod
    def _resolve_point_color(config: Mapping[str, Any], agent_id, role: str, default_color, ego_color):
        lidar_point_colors = config["lidar_point_colors"]
        if agent_id is not None and agent_id in lidar_point_colors:
            return CoperceptionVisualizer._as_uint8_color(lidar_point_colors[agent_id])
        if role == "ego":
            return ego_color
        return default_color

    @staticmethod
    def _to_numpy_array(value):
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)

        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        return np.array(value, copy=True)


if TYPE_CHECKING:
    DatasetOpenCOOD: TypeAlias = LateFusionDataset | EarlyFusionDataset | IntermediateFusionDataset | IntermediateFusionDatasetV2
else:
    DatasetOpenCOOD: TypeAlias = object


class CoperceptionModelManager:
    def __init__(self, opt, current_time, payload_handler=None, visualization_config=None):
        self.opt = opt
        self.hypes = yaml_utils.load_yaml(None, self.opt)
        self.model = train_utils.create_model(self.hypes)
        self.current_time = current_time
        self.vis = None
        self.visualization_config = CoperceptionVisualizer.resolve_visualization_config(visualization_config)

        if torch.cuda.is_available():
            self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saved_path = self.opt.model_dir
        _, self.model = train_utils.load_saved_model(self.saved_path, self.model)

        self.opencood_dataset: DatasetOpenCOOD | None = None
        self.data_loader: DataLoader[Any] | None = None
        self.payload_handler = payload_handler
        self.current_memory_data = None
        self.advcp_config = load_advcp_config(getattr(self.opt, "advcp_config", None)) if getattr(self.opt, "with_advcp", False) else None

        logger.info("Initial Dataset Building")
        self.opencood_dataset = cast(DatasetOpenCOOD, build_dataset(self.hypes, visualize=True, train=False, payload_handler=self.payload_handler))

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

    def update_dataset(self, data=None):
        logger.debug("Refreshing dataset indices")
        self.current_memory_data = data
        self.opencood_dataset.update_database(memory_data=data)

        if len(self.opencood_dataset) == 0:
            logger.warning("No samples found in dataset after update.")

    def validate_advcp_agents(self, valid_agent_ids):
        if not getattr(self.opt, "with_advcp", False) or self.advcp_config is None:
            return

        resolved_attacker_id = resolve_attacker_id(self.advcp_config.get("attacker_id"), list(valid_agent_ids))
        self.advcp_config["attacker_id"] = resolved_attacker_id

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
                self.vis.get_render_option().background_color = (  # noqa: DC05
                    np.asarray(self.visualization_config["background"], dtype=np.float64) / 255.0
                )  # noqa: DC05
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
                batch_data = train_utils.to_device(batch_data, self.device)
                if self.opt.fusion_method == "late":
                    if getattr(self.opt, "with_advcp", False):
                        pred_box_tensor, pred_score, gt_box_tensor = inference_late_fusion_attack(
                            batch_data,
                            self.model,
                            self.opencood_dataset,
                            self.device,
                            advcp_config=self.advcp_config,
                            memory_data=self.current_memory_data,
                        )
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                elif self.opt.fusion_method == "early":
                    if getattr(self.opt, "with_advcp", False):
                        pred_box_tensor, pred_score, gt_box_tensor = inference_early_fusion_attack(batch_data, self.model, self.opencood_dataset)
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(
                            batch_data, self.model, self.opencood_dataset
                        )
                elif self.opt.fusion_method == "intermediate":
                    if getattr(self.opt, "with_advcp", False):
                        pred_box_tensor, pred_score, gt_box_tensor = inference_intermediate_fusion_attack(
                            batch_data, self.model, self.opencood_dataset
                        )
                    else:
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
                        pcd_points = None
                        ego_data = batch_data["ego"]
                        if "origin_lidar" in ego_data:
                            pcd_points = ego_data["origin_lidar"]
                            if self.hypes.get("fusion", {}).get("core_method") == "IntermediateFusionDatasetV2":
                                pcd_points = pcd_points[:, 1:]
                            if isinstance(pcd_points, list) or (hasattr(pcd_points, "ndim") and pcd_points.ndim > 2):
                                pcd_points = pcd_points[0]
                        elif "lidar_np" in ego_data:
                            pcd_points = ego_data["lidar_np"]
                            if isinstance(pcd_points, list):
                                pcd_points = pcd_points[0]
                        vis_dir = f"simulation_output/coperception/vis_{mode}/{self.opt.test_scenario}_{self.current_time}"
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_save_path = os.path.join(vis_dir, f"{mode}_{tick_number:05d}.png")
                        CoperceptionVisualizer.render_inference_to_file(
                            pred_box_tensor,
                            gt_box_tensor,
                            pcd_points,
                            self.hypes["postprocess"]["gt_range"],
                            vis_save_path,
                            batch_data=batch_data,
                            visualization_config=self.visualization_config,
                            method=mode,
                            left_hand=True,
                            vis_pred_box=True,
                        )

                if self.opt.show_vis:
                    if self.hypes["postprocess"]["core_method"] == "BevPostprocessor":
                        CoperceptionVisualizer.show_inference_image(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"],
                            self.hypes["postprocess"]["gt_range"],
                            batch_data=batch_data,
                            visualization_config=self.visualization_config,
                            method="bev",
                            left_hand=True,
                        )
                    else:
                        CoperceptionVisualizer.show_inference_image(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"],
                            self.hypes["postprocess"]["gt_range"],
                            batch_data=batch_data,
                            visualization_config=self.visualization_config,
                            method="3d",
                            left_hand=True,
                        )

                if self.opt.show_sequence and pred_box_tensor is not None and self.hypes["postprocess"]["core_method"] != "BevPostprocessor":
                    self.vis.clear_geometries()
                    pcd, pred_o3d_box, gt_o3d_box = CoperceptionVisualizer.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data["ego"]["origin_lidar"],
                        vis_pcd,
                        batch_data=batch_data,
                        visualization_config=self.visualization_config,
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

    def final_eval(self):
        eval_dir = f"simulation_output/coperception/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)
        eval_utils.eval_final_results(self.final_result_stat, eval_dir, self.opt.global_sort_detections)


class DirectoryProcessor:
    def __init__(self, source_directory="data_dumping", max_cav=None):
        self.source_directory = source_directory
        self.max_cav = int(max_cav) if max_cav is not None else None

    def detect_cameras(self, data_directory):
        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])
        if not inner_subdirectories:
            return []

        sample_folder = os.path.join(data_directory, inner_subdirectories[0])
        camera_files = [f for f in os.listdir(sample_folder) if re.match(r"\d+_camera\d+\.png", f)]

        camera_ids = sorted(set(re.findall(r"_camera(\d+)\.png", f)[0] for f in camera_files if re.findall(r"_camera(\d+)\.png", f)))

        return [f"_camera{cam_id}.png" for cam_id in camera_ids]

    def retrieve_data_structure(self, tick_number):
        number = f"{tick_number:06d}"

        subdirectories = sorted([d for d in os.listdir(self.source_directory) if os.path.isdir(os.path.join(self.source_directory, d))])

        if len(subdirectories) < 2:
            return None

        data_directory = os.path.join(self.source_directory, subdirectories[-2])

        try:
            camera_postfixes = self.detect_cameras(data_directory)
        except Exception:
            camera_postfixes = []

        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])

        if not inner_subdirectories:
            return None

        if "rsu" in inner_subdirectories[0]:
            inner_subdirectories = inner_subdirectories[1:] + [inner_subdirectories[0]]

        if self.max_cav is not None and self.max_cav > 0 and len(inner_subdirectories) > self.max_cav:
            logger.warning(f"Too many CAVs and RSUs: {len(inner_subdirectories)}")
            logger.warning(f"Maximum is {self.max_cav}")
            inner_subdirectories = inner_subdirectories[: self.max_cav]

        expected_ego_id = inner_subdirectories[0]
        expected_agent_path = os.path.join(data_directory, expected_ego_id)
        expected_yaml_path = os.path.join(expected_agent_path, f"{number}.yaml")
        expected_lidar_path = os.path.join(expected_agent_path, f"{number}.pcd")
        if not os.path.exists(expected_yaml_path) or not os.path.exists(expected_lidar_path):
            logger.warning(f"Skipping tick {tick_number}: expected ego agent '{expected_ego_id}' has incomplete data.")
            return None

        scenario_data = OrderedDict()
        scenario_data[0] = OrderedDict()

        agents_found_count = 0

        for j, folder in enumerate(inner_subdirectories):
            cav_id = folder
            agent_path = os.path.join(data_directory, cav_id)

            yaml_path = os.path.join(agent_path, f"{number}.yaml")
            lidar_path = os.path.join(agent_path, f"{number}.pcd")

            if not os.path.exists(yaml_path) or not os.path.exists(lidar_path):
                continue

            scenario_data[0][cav_id] = OrderedDict()
            timestamp = number
            scenario_data[0][cav_id][timestamp] = OrderedDict()

            scenario_data[0][cav_id][timestamp]["yaml"] = yaml_path
            scenario_data[0][cav_id][timestamp]["lidar"] = lidar_path

            camera_files = []
            for postfix in camera_postfixes:
                cam_path = os.path.join(agent_path, f"{number}{postfix}")

                if os.path.exists(cam_path):
                    camera_files.append(cam_path)
                else:
                    pass

            scenario_data[0][cav_id][timestamp]["camera0"] = camera_files

            scenario_data[0][cav_id]["ego"] = cav_id == expected_ego_id

            agents_found_count += 1

        if agents_found_count == 0:
            return None

        return scenario_data
