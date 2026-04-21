from __future__ import annotations

import copy
import os
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Tuple, TypeAlias, TypedDict, cast

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

if TYPE_CHECKING:
    from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset

    DatasetOpenCOOD: TypeAlias = LateFusionDataset | EarlyFusionDataset | IntermediateFusionDataset | IntermediateFusionDatasetV2
else:
    DatasetOpenCOOD: TypeAlias = object

logger = logging.getLogger("cavise.opencda.opencda.core.common.coperception_model_manager")

ColorRGB = tuple[int, int, int]
CoreMethodName = Literal[
    "LateFusionDataset",
    "EarlyFusionDataset",
    "IntermediateFusionDataset",
    "IntermediateFusionDatasetV2",
]


class CoperceptionVisualizationConfig(TypedDict):
    background: ColorRGB  # noqa: DC01
    lidar_point_colors: dict[str, ColorRGB]
    bbox_colors: dict[str, ColorRGB]  # noqa: DC01
    bbox_line_thickness: int  # noqa: DC01
    image_dpi: int  # noqa: DC01


@dataclass
class CoperceptionInferenceResult:
    pred_box_tensor: torch.Tensor | None
    pred_score: torch.Tensor | None
    gt_box_tensor: torch.Tensor | None
    visualization_context: Optional[Mapping[str, Any]] = None


class InferenceMapper:
    _INFERENCE_CORE_METHOD_ATTR = "__inference_core_method__"
    mapping: dict[CoreMethodName, Callable[..., CoperceptionInferenceResult]] = {}

    @classmethod
    def for_core_method(
        cls,
        core_method: CoreMethodName,
    ) -> Callable[[Callable[..., CoperceptionInferenceResult]], Callable[..., CoperceptionInferenceResult]]:
        def decorator(
            func: Callable[..., CoperceptionInferenceResult],
        ) -> Callable[..., CoperceptionInferenceResult]:
            registered_methods = tuple(getattr(func, cls._INFERENCE_CORE_METHOD_ATTR, ()))
            setattr(func, cls._INFERENCE_CORE_METHOD_ATTR, (*registered_methods, core_method))
            cls.mapping[core_method] = func
            return func

        return decorator

    @classmethod
    def resolve(
        cls,
        owner: type["CoperceptionModelManager"],
        core_method: str | None,
    ) -> Callable[..., CoperceptionInferenceResult]:
        for base in owner.__mro__:
            for value in base.__dict__.values():
                if core_method in getattr(value, cls._INFERENCE_CORE_METHOD_ATTR, ()):
                    return cast(Callable[..., CoperceptionInferenceResult], getattr(owner, value.__name__))

        supported_methods = ", ".join(cls.mapping.keys())
        raise NotImplementedError(
            f'Unsupported cooperative perception fusion.core_method "{core_method}". Supported core methods: {supported_methods}.'
        )


@dataclass
class IoUResultStat:
    tp: list[Any]
    fp: list[Any]
    gt: int
    score: list[Any]

    @classmethod
    def create_empty(cls) -> "IoUResultStat":
        return cls(tp=[], fp=[], gt=0, score=[])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "gt": self.gt,
            "score": self.score,
        }

    def merge_from(self, other: "IoUResultStat") -> None:
        self.gt += other.gt
        self.tp += other.tp
        self.fp += other.fp
        self.score += other.score


@dataclass
class EvaluationResultStat:
    by_iou: Dict[float, IoUResultStat]

    IOU_THRESHOLDS = (0.3, 0.5, 0.7)

    @classmethod
    def create_empty(cls) -> "EvaluationResultStat":
        return cls({iou: IoUResultStat.create_empty() for iou in cls.IOU_THRESHOLDS})

    def __getitem__(self, iou: float) -> IoUResultStat:
        return self.by_iou[iou]

    def items(self):
        return self.by_iou.items()

    def as_dict(self) -> Dict[float, Dict[str, Any]]:
        return {iou: stat.as_dict() for iou, stat in self.by_iou.items()}

    def merge_from(self, other: "EvaluationResultStat") -> None:
        for iou in self.IOU_THRESHOLDS:
            self.by_iou[iou].merge_from(other[iou])


@dataclass
class SequenceVisualizationState:
    pcd: Any
    box_groups: Dict[str, list[Any]]


class CoperceptionVisualizer:
    _DEFAULT_VISUALIZATION_CONFIG: CoperceptionVisualizationConfig = {
        "background": (0, 0, 0),
        "lidar_point_colors": {
            "other": (255, 255, 255),
            "ego": (80, 255, 80),
        },
        "bbox_colors": {
            "gt": (0, 255, 0),
            "pred": (255, 0, 0),
        },
        "bbox_line_thickness": 5,
        "image_dpi": 400,
    }

    @classmethod
    def resolve_visualization_config(cls, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        resolved = copy.deepcopy(cls._DEFAULT_VISUALIZATION_CONFIG)
        if not config:
            return resolved

        config_dict = dict(config)
        for key in ("background",):
            if key in config_dict and config_dict[key] is not None:
                resolved[key] = tuple(config_dict[key])

        for key in ("bbox_line_thickness", "image_dpi"):
            if key in config_dict and config_dict[key] is not None:
                resolved[key] = int(config_dict[key])

        for key in ("lidar_point_colors", "bbox_colors"):
            value = config_dict.get(key)
            if isinstance(value, Mapping):
                normalized_values = {name: tuple(color) for name, color in dict(value).items()}
                if key == "lidar_point_colors" and "default" in normalized_values and "other" not in normalized_values:
                    normalized_values["other"] = normalized_values["default"]
                resolved[key].update(normalized_values)

        return resolved

    @classmethod
    def render_inference_to_file(
        cls,
        pred_box_tensor,
        gt_tensor,
        pcd,
        pc_range,
        save_path,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
        visualization_context: Optional[Mapping[str, Any]] = None,
        method: str = "3d",
        vis_gt_box: bool = True,
        vis_pred_box: bool = True,
        left_hand: bool = False,
        uncertainty=None,
    ):
        config = cls.resolve_visualization_config(visualization_config)
        canvas = cls._build_canvas(
            pred_box_tensor=pred_box_tensor,
            gt_tensor=gt_tensor,
            pcd=pcd,
            pc_range=pc_range,
            batch_data=batch_data,
            visualization_config=config,
            visualization_context=visualization_context,
            method=method,
            vis_gt_box=vis_gt_box,
            vis_pred_box=vis_pred_box,
            left_hand=left_hand,
            uncertainty=uncertainty,
        )
        plt.axis("off")
        plt.imshow(canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=config["image_dpi"], pad_inches=0.0)
        plt.clf()

    @classmethod
    def visualize_inference_sample_dataloader(
        cls,
        pred_box_tensor,
        gt_box_tensor,
        origin_lidar,
        o3d_pcd,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
        visualization_context: Optional[Mapping[str, Any]] = None,
    ):
        config = cls.resolve_visualization_config(visualization_config)
        origin_lidar_np, point_colors_uint8 = cls._get_lidar_points_and_colors(
            batch_data,
            origin_lidar,
            config,
            visualization_context=visualization_context,
        )
        pred_color = cls._as_float_color(config["bbox_colors"]["pred"])
        gt_color = cls._as_float_color(config["bbox_colors"]["gt"])

        origin_lidar_np = np.array(origin_lidar_np, copy=True)
        origin_lidar_np[:, :1] = -origin_lidar_np[:, :1]
        point_colors = point_colors_uint8.astype(np.float64) / 255.0
        o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar_np[:, :3])
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_colors)

        box_groups = {
            "pred": vis_utils.bbx2linset(pred_box_tensor, color=pred_color) if pred_box_tensor is not None else [],
            "gt": vis_utils.bbx2linset(gt_box_tensor, color=gt_color) if gt_box_tensor is not None else [],
        }
        for box_name, box_tensor in cls._get_extra_box_tensors(visualization_context).items():
            if box_tensor is None:
                box_groups[box_name] = []
                continue
            box_groups[box_name] = vis_utils.bbx2linset(box_tensor, color=cls._as_float_color(config["bbox_colors"][box_name]))

        return o3d_pcd, box_groups

    @classmethod
    def _build_canvas(
        cls,
        pred_box_tensor,
        gt_tensor,
        pcd,
        pc_range,
        batch_data=None,
        visualization_config: Optional[Mapping[str, Any]] = None,
        visualization_context: Optional[Mapping[str, Any]] = None,
        method: str = "3d",
        vis_gt_box: bool = True,
        vis_pred_box: bool = True,
        left_hand: bool = False,
        uncertainty=None,
    ):
        import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
        import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
        from opencood.utils import common_utils

        config = cls.resolve_visualization_config(visualization_config)
        pc_range = [int(i) for i in pc_range]
        pcd_np, point_colors = cls._get_lidar_points_and_colors(
            batch_data,
            pcd,
            config,
            visualization_context=visualization_context,
        )
        bg_color = cls._as_uint8_color(config["background"])
        gt_color = cls._as_uint8_color(config["bbox_colors"]["gt"])
        pred_color = cls._as_uint8_color(config["bbox_colors"]["pred"])
        box_line_thickness = int(config["bbox_line_thickness"])

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
                canvas.draw_boxes(gt_box_np, colors=gt_color, texts=gt_name, box_line_thickness=box_line_thickness)
            if vis_pred_box and pred_box_np is not None:
                canvas.draw_boxes(pred_box_np, colors=pred_color, texts=pred_name, box_line_thickness=box_line_thickness)
            cls._draw_extra_boxes(canvas, method, config, visualization_context, common_utils)
        elif method == "3d":
            canvas = canvas_3d.Canvas_3D(canvas_bg_color=bg_color, left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            if valid_mask.any():
                canvas.draw_canvas_points(canvas_xy[valid_mask], colors=point_colors[valid_mask])
            if vis_gt_box and gt_box_np is not None:
                canvas.draw_boxes(gt_box_np, colors=gt_color, texts=gt_name, box_line_thickness=box_line_thickness)
            if vis_pred_box and pred_box_np is not None:
                canvas.draw_boxes(pred_box_np, colors=pred_color, texts=pred_name, box_line_thickness=box_line_thickness)
            cls._draw_extra_boxes(canvas, method, config, visualization_context, common_utils)
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

    @classmethod
    def _get_lidar_points_and_colors(
        cls,
        batch_data,
        fallback_pcd,
        config: Mapping[str, Any],
        visualization_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        other_color = cls._as_uint8_color(config["lidar_point_colors"]["other"])
        ego_color = cls._as_uint8_color(config["lidar_point_colors"].get("ego", other_color))

        if isinstance(batch_data, Mapping):
            ego_entry = batch_data.get("ego")
            if isinstance(ego_entry, Mapping) and "origin_lidar_by_agent" in ego_entry:
                points_by_agent = ego_entry["origin_lidar_by_agent"]
                roles = list(ego_entry.get("origin_lidar_roles", []))
                agent_ids = list(ego_entry.get("origin_lidar_agent_ids", []))
                colored_points = []
                colored_values = []

                for idx, points in enumerate(points_by_agent):
                    agent_points = cls._to_numpy_points(points)
                    if agent_points.size == 0:
                        continue

                    role = roles[idx] if idx < len(roles) else "default"
                    agent_id = agent_ids[idx] if idx < len(agent_ids) else None
                    color = cls._resolve_point_color(
                        config,
                        agent_id=agent_id,
                        role=role,
                        other_color=other_color,
                        ego_color=ego_color,
                        visualization_context=visualization_context,
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

                cav_points = cls._to_numpy_points(local_lidar)
                if cav_points.size == 0:
                    continue

                if cav_id != "ego":
                    from opencood.utils import box_utils

                    transformation_matrix = cav_content.get("transformation_matrix")
                    if transformation_matrix is not None:
                        transformation_matrix_np = cls._to_numpy_array(transformation_matrix)
                        cav_points[:, :3] = box_utils.project_points_by_matrix_torch(cav_points[:, :3], transformation_matrix_np)

                cav_color = cls._resolve_point_color(
                    config,
                    agent_id=cav_id,
                    role="ego" if cav_id == "ego" else "default",
                    other_color=other_color,
                    ego_color=ego_color,
                    visualization_context=visualization_context,
                )
                colored_points.append(cav_points)
                colored_values.append(np.tile(np.asarray(cav_color, dtype=np.uint8), (cav_points.shape[0], 1)))

            if colored_points:
                return np.vstack(colored_points), np.vstack(colored_values)

        fallback_points = cls._to_numpy_points(fallback_pcd)
        fallback_colors = np.tile(np.asarray(other_color, dtype=np.uint8), (fallback_points.shape[0], 1))
        return fallback_points, fallback_colors

    @staticmethod
    def _as_uint8_color(color: Iterable[Any]) -> Tuple[int, int, int]:
        rgb = [int(v) for v in list(color)[:3]]
        return tuple(max(0, min(255, value)) for value in rgb)

    @staticmethod
    def _as_float_color(color: Iterable[Any]) -> Tuple[float, float, float]:
        return tuple(channel / 255.0 for channel in CoperceptionVisualizer._as_uint8_color(color))

    @classmethod
    def _resolve_point_color(
        cls,
        config: Mapping[str, Any],
        agent_id: str,
        role: str,
        other_color: tuple,
        ego_color: tuple,
        visualization_context: Optional[Mapping[str, Any]] = None,
    ):
        lidar_point_colors = config["lidar_point_colors"]
        if agent_id is not None and agent_id in lidar_point_colors:
            return cls._as_uint8_color(lidar_point_colors[agent_id])
        if role == "ego":
            return ego_color
        return other_color

    @classmethod
    def _get_extra_box_tensors(cls, visualization_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return {}

    @classmethod
    def _draw_extra_boxes(cls, canvas, method: str, config: Mapping[str, Any], visualization_context, common_utils):
        for box_name, box_tensor in cls._get_extra_box_tensors(visualization_context).items():
            if box_tensor is None:
                continue
            box_np = common_utils.torch_tensor_to_numpy(box_tensor)
            color = cls._as_uint8_color(config["bbox_colors"][box_name])
            texts = [""] * box_np.shape[0]
            box_line_thickness = int(config["bbox_line_thickness"])
            if method == "bev":
                canvas.draw_boxes(box_np, colors=color, texts=texts, box_line_thickness=box_line_thickness)
            else:
                canvas.draw_boxes(box_np, colors=color, texts=texts, box_line_thickness=box_line_thickness)

    @staticmethod
    def _to_numpy_array(value):
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)

        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()

        return np.array(value, copy=True)


class CoperceptionModelManager:
    VISUALIZER_CLASS = CoperceptionVisualizer
    SEQUENCE_BOX_GROUP_NAMES: tuple[str, ...] = ("pred", "gt")

    def __init__(self, opt, current_time, payload_handler=None, visualization_config=None):
        self.opt = opt
        self.hypes = yaml_utils.load_yaml(None, self.opt)
        self.current_time = current_time
        self.vis = None
        self.visualization_config = CoperceptionVisualizer.resolve_visualization_config(visualization_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saved_path = self.opt.model_dir
        self.model = self._init_model()
        self.opencood_dataset: DatasetOpenCOOD | None = None
        self.data_loader: DataLoader[Any] | None = None
        self.current_memory_data = None
        self.payload_handler = payload_handler
        self.inference = self._select_inference()

        self._init_dataset()
        self.final_result_stat = EvaluationResultStat.create_empty()

    def _init_model(self):
        model = train_utils.create_model(self.hypes)
        if torch.cuda.is_available():
            model.cuda()
        _, model = train_utils.load_saved_model(self.saved_path, model)
        return model

    def _init_dataset(self) -> None:
        logger.info("Initial Dataset Building")
        self.opencood_dataset = cast(DatasetOpenCOOD, build_dataset(self.hypes, visualize=True, train=False, payload_handler=self.payload_handler))
        self.data_loader = self._create_data_loader(self.opencood_dataset)

    @staticmethod
    def _create_data_loader(dataset) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def update_dataset(self, data=None):
        logger.debug("Refreshing dataset indices")
        self.current_memory_data = data
        self.opencood_dataset.update_database(memory_data=data)

        if len(self.opencood_dataset) == 0:
            logger.warning("No samples found in dataset after update.")

    def _resolve_inference_callable(self) -> Callable[..., CoperceptionInferenceResult]:
        core_method = self.hypes.get("fusion", {}).get("core_method")
        return InferenceMapper.resolve(type(self), core_method)

    def _select_inference(self) -> Callable[[Any], CoperceptionInferenceResult]:
        inference_callable = self._resolve_inference_callable()
        return cast(Callable[[Any], CoperceptionInferenceResult], inference_callable.__get__(self, type(self)))

    @InferenceMapper.for_core_method("LateFusionDataset")  # noqa: DC04
    def _run_late_inference(self, batch_data):
        return self._build_inference_result(*inference_utils.inference_late_fusion(batch_data, self.model, self.opencood_dataset))

    @InferenceMapper.for_core_method("EarlyFusionDataset")  # noqa: DC04
    def _run_early_inference(self, batch_data):
        return self._build_inference_result(*inference_utils.inference_early_fusion(batch_data, self.model, self.opencood_dataset))

    @InferenceMapper.for_core_method("IntermediateFusionDataset")  # noqa: DC04
    @InferenceMapper.for_core_method("IntermediateFusionDatasetV2")
    def _run_intermediate_inference(self, batch_data):
        return self._build_inference_result(*inference_utils.inference_intermediate_fusion(batch_data, self.model, self.opencood_dataset))

    @staticmethod
    def _build_inference_result(*inference_output) -> CoperceptionInferenceResult:
        if len(inference_output) == 3:
            pred_box_tensor, pred_score, gt_box_tensor = inference_output
            visualization_context = None
        elif len(inference_output) == 4:
            pred_box_tensor, pred_score, gt_box_tensor, visualization_context = inference_output
        else:
            raise ValueError("Inference output must contain 3 or 4 elements.")

        return CoperceptionInferenceResult(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            visualization_context,
        )

    @staticmethod
    def _create_evaluation_stat() -> EvaluationResultStat:
        return EvaluationResultStat.create_empty()

    @staticmethod
    def _update_evaluation_stat(
        result_stat: EvaluationResultStat,
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
    ) -> None:
        for iou in EvaluationResultStat.IOU_THRESHOLDS:
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou)

    def _extract_visualization_pcd(self, batch_data):
        ego_data = batch_data["ego"]
        if "origin_lidar" in ego_data:
            pcd_points = ego_data["origin_lidar"]
            if isinstance(pcd_points, list) or (hasattr(pcd_points, "ndim") and pcd_points.ndim > 2):
                pcd_points = pcd_points[0]
            if self.hypes.get("fusion", {}).get("core_method") == "IntermediateFusionDatasetV2":
                pcd_points = pcd_points[:, 1:]
            return pcd_points

        if "lidar_np" in ego_data:
            pcd_points = ego_data["lidar_np"]
            if isinstance(pcd_points, list):
                pcd_points = pcd_points[0]
            return pcd_points

        return None

    def _save_prediction_npy(self, pred_box_tensor, gt_box_tensor, batch_data, index: int) -> None:
        npy_dir = f"simulation_output/coperception/npy/{self.opt.test_scenario}_{self.current_time}"
        npy_save_path = os.path.join(npy_dir, "npy")
        os.makedirs(npy_save_path, exist_ok=True)
        inference_utils.save_prediction_gt(
            pred_box_tensor,
            gt_box_tensor,
            batch_data["ego"]["origin_lidar"][0],
            index,
            npy_save_path,
        )

    def _save_visualizations(
        self,
        visualizer_cls,
        pred_box_tensor,
        gt_box_tensor,
        batch_data,
        tick_number: int,
        visualization_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        pcd_points = self._extract_visualization_pcd(batch_data)
        for mode in ("3d", "bev"):
            if self.hypes["postprocess"]["core_method"] == "BevPostprocessor" and mode == "3d":
                continue
            vis_dir = f"simulation_output/coperception/vis_{mode}/{self.opt.test_scenario}_{self.current_time}"
            os.makedirs(vis_dir, exist_ok=True)
            vis_save_path = os.path.join(vis_dir, f"{mode}_{tick_number:05d}.png")
            visualizer_cls.render_inference_to_file(
                pred_box_tensor,
                gt_box_tensor,
                pcd_points,
                self.hypes["postprocess"]["gt_range"],
                vis_save_path,
                batch_data=batch_data,
                visualization_config=self.visualization_config,
                visualization_context=visualization_context,
                method=mode,
                left_hand=True,
                vis_pred_box=True,
            )

    def _init_sequence_visualization(self) -> SequenceVisualizationState:
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()  # noqa: DC05
            self.vis.create_window()  # noqa: DC05
            self.vis.get_render_option().background_color = (  # noqa: DC05
                np.asarray(self.visualization_config["background"], dtype=np.float64) / 255.0
            )  # noqa: DC05
            self.vis.get_render_option().point_size = 1.0  # noqa: DC05
            self.vis.get_render_option().show_coordinate_frame = True  # noqa: DC05
        return SequenceVisualizationState(
            pcd=o3d.geometry.PointCloud(),
            box_groups={group_name: [] for group_name in self.SEQUENCE_BOX_GROUP_NAMES},
        )

    def _update_sequence_visualization(
        self,
        sequence_state: SequenceVisualizationState,
        visualizer_cls,
        pred_box_tensor,
        gt_box_tensor,
        batch_data,
        visualization_context: Optional[Mapping[str, Any]],
        index: int,
    ) -> None:
        if pred_box_tensor is None or self.hypes["postprocess"]["core_method"] == "BevPostprocessor":
            return

        self.vis.clear_geometries()
        pcd, box_groups = visualizer_cls.visualize_inference_sample_dataloader(
            pred_box_tensor,
            gt_box_tensor,
            batch_data["ego"]["origin_lidar"],
            sequence_state.pcd,
            batch_data=batch_data,
            visualization_config=self.visualization_config,
            visualization_context=visualization_context,
        )
        self.vis.add_geometry(pcd)

        for group_name in self.SEQUENCE_BOX_GROUP_NAMES:
            line_sets = list(box_groups.get(group_name, []))
            sequence_state.box_groups[group_name] = line_sets
            for line_set in line_sets:
                self.vis.add_geometry(line_set)

        self.vis.update_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def make_prediction(self, tick_number):
        self.model.eval()
        result_stat = self._create_evaluation_stat()
        sequence_state = self._init_sequence_visualization() if self.opt.show_video_vis else None

        visualizer_cls = self.VISUALIZER_CLASS
        inference_name = getattr(self.inference, "__qualname__", repr(self.inference))
        logger.info("Using cooperative perception inference: %s", inference_name)
        batch_count = len(self.data_loader)

        if batch_count != 1:
            logger.warning(
                "Expected exactly 1 batch in cooperative perception data loader, got %s.",
                batch_count,
            )
            if batch_count == 0:
                logger.warning("Skipping cooperative perception prediction because the data loader is empty.")
                return
            logger.warning("Only the first batch will be processed.")

        batch_data = next(iter(self.data_loader))
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            inference_result = self.inference(batch_data)
            pred_box_tensor = inference_result.pred_box_tensor
            pred_score = inference_result.pred_score
            gt_box_tensor = inference_result.gt_box_tensor
            visualization_context = inference_result.visualization_context

            self._update_evaluation_stat(result_stat, pred_box_tensor, pred_score, gt_box_tensor)

            if self.opt.save_npy:
                self._save_prediction_npy(pred_box_tensor, gt_box_tensor, batch_data, 0)

            if self.opt.save_vis:
                self._save_visualizations(
                    visualizer_cls,
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data,
                    tick_number,
                    visualization_context=visualization_context,
                )

            if sequence_state is not None:
                self._update_sequence_visualization(
                    sequence_state,
                    visualizer_cls,
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data,
                    visualization_context,
                    0,
                )

        self.final_result_stat.merge_from(result_stat)

    def final_eval(self):
        eval_dir = f"simulation_output/coperception/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)
        eval_utils.eval_final_results(self.final_result_stat.as_dict(), eval_dir, self.opt.global_sort_detections)
