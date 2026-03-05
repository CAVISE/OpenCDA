from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .general import get_xylims, draw_bbox_2d, draw_bboxes_2d, draw_polygons, draw_pointclouds, show_or_save
from mvp.data.util import pcd_sensor_to_map, bbox_sensor_to_map
from mvp.config import color_map


def draw_ground_segmentation(pcd_data: np.ndarray, inliers: np.ndarray, show: bool = False, save: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(30, 30))
    pointcloud = pcd_data[:, :3]
    xlim, ylim = get_xylims(pointcloud)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], s=0.01, c="black")
    ax.scatter(pointcloud[inliers, 0], pointcloud[inliers, 1], s=0.01, c="red")
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def draw_sensor_calib(
    pcd_on_camera: Optional[np.ndarray],
    camera_image: np.ndarray,
    camera_seg: Optional[Dict[str, Any]] = None,
    lidar_seg: Optional[Dict[str, Any]] = None,
    show: bool = False,
    save: Optional[str] = None,
) -> None:
    image_shape = camera_image.shape
    fig, ax = plt.subplots()

    image_extent: Tuple[float, float, float, float] = (0, image_shape[1], image_shape[0], 0)
    if camera_seg is not None:
        camera_image = camera_seg["img"]
    ax.imshow(camera_image, origin="upper", extent=image_extent)

    if pcd_on_camera is not None:
        in_screen_mask = pcd_on_camera[:, 2] > 0
        if lidar_seg is not None:
            # for i, info in enumerate(lidar_seg["info"]):
            #     points = pcd_on_camera[info["indices"]]
            #     ax.scatter(points[:,0], points[:,1], s=0.02, color=[0,0,(i+1)/17])
            classes = np.unique(lidar_seg["class"])
            for class_id in classes.tolist():
                class_mask = lidar_seg["class"] == class_id
                indices = np.argwhere(in_screen_mask * class_mask > 0).reshape(-1)
                points = pcd_on_camera[indices, :]
                ax.scatter(points[:, 0], points[:, 1], s=0.02, color=(np.array(color_map[class_id]) / 255).tolist())
        else:
            ax.scatter(pcd_on_camera[:, 0], pcd_on_camera[:, 1], s=0.02, c="blue")
    ax.set_xlim((image_extent[0], image_extent[1]))
    ax.set_ylim((image_extent[2], image_extent[3]))

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def draw_polygon_areas(case: Dict[Any, Any], show: bool = False, save: Optional[str] = None, tag: str = "") -> None:
    fig, ax = plt.subplots(figsize=(30, 30))
    color_map_local = ["r", "g", "b", "k", "y"]

    for i, vehicle_id in enumerate(case):
        vehicle_data = case[vehicle_id]

        if "lidar" in vehicle_data and vehicle_data["lidar"] is not None:
            lidar = vehicle_data["lidar"]
            lidar_pose = vehicle_data["lidar_pose"]
            pcd = pcd_sensor_to_map(lidar, lidar_pose)
            plt.scatter(pcd[:, 0], pcd[:, 1], s=0.1, c=color_map_local[i])

        if "gt_bboxes" in vehicle_data:
            bboxes_to_draw: List[Tuple[np.ndarray, Optional[List[Any]], str]] = [
                (bbox_sensor_to_map(vehicle_data["gt_bboxes"], vehicle_data["lidar_pose"]), None, "g")
            ]
            draw_bbox_2d(ax, bboxes_to_draw)

        if "pred_bboxes" in vehicle_data:
            bboxes_to_draw_pred: List[Tuple[np.ndarray, Optional[List[Any]], str]] = [
                (bbox_sensor_to_map(vehicle_data["pred_bboxes"], vehicle_data["lidar_pose"]), None, color_map_local[i])
            ]
            draw_bbox_2d(ax, bboxes_to_draw_pred)

        if "free_areas" + tag in vehicle_data:
            for area in vehicle_data["free_areas" + tag]:
                x, y = area.exterior.coords.xy
                plt.fill(x, y, color_map_local[i], alpha=0.2)
        if "occupied_areas" + tag in vehicle_data:
            for area in vehicle_data["occupied_areas" + tag]:
                x, y = area.exterior.coords.xy
                plt.plot(x, y, color_map_local[i], alpha=0.8)
        if "ego_area" in vehicle_data:
            area = vehicle_data["ego_area"]
            x, y = area.exterior.coords.xy
            plt.plot(x, y, color_map_local[i])
            plt.text(x[0], y[0], str(vehicle_id))

    ax.set_aspect("equal", adjustable="box")

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_object_tracking(
    point_clouds: List[np.ndarray],
    detections: List[np.ndarray],
    predictions: List[np.ndarray],
    show: bool = False,
    save: Optional[str] = None,
) -> None:
    frame_num = len(detections)
    fig, axes = plt.subplots(frame_num, 1, figsize=(10, 10 * frame_num))

    for frame_id in range(frame_num):
        point_cloud = point_clouds[frame_id]
        ax = axes[frame_id]
        xlim, ylim = get_xylims(point_cloud)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.01, c="black")
        ax.scatter(predictions[frame_id][:, 0], predictions[frame_id][:, 1], s=50, c="red")
        ax.scatter(detections[frame_id][:, 0], detections[frame_id][:, 1], s=50, c="blue")

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def visualize_defense(case: List[Dict[Any, Any]], metrics: List[Dict[Any, Any]], show: bool = False, save: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(30, 30))
    vehicle_color_map = ["r", "g", "b", "k", "y"]

    frame_data = case[-1]
    vehicle_ids = list(frame_data.keys())

    for i, vehicle_id in enumerate(vehicle_ids):
        if frame_data[vehicle_id]["lidar"] is not None:
            draw_pointclouds(ax, pcd_sensor_to_map(frame_data[vehicle_id]["lidar"], frame_data[vehicle_id]["lidar_pose"]), color=vehicle_color_map[i])
        draw_polygons(ax, frame_data[vehicle_id]["free_areas"], color=vehicle_color_map[i], alpha=0.2, border=False)
        draw_polygons(ax, frame_data[vehicle_id]["occupied_areas"], color=vehicle_color_map[i], alpha=0.6, fill=True, border=False, linewidth=2)
        draw_polygons(ax, frame_data[vehicle_id]["ego_area"], color=vehicle_color_map[i], alpha=0.6, fill=True, border=False, linewidth=2)
        draw_bboxes_2d(ax, frame_data[vehicle_id]["ego_bbox"][np.newaxis, :], None, color="g", linewidth=5)

    for i, vehicle_id in enumerate(vehicle_ids):
        if "pred_bboxes" in frame_data[vehicle_id]:
            draw_bboxes_2d(
                ax, bbox_sensor_to_map(frame_data[vehicle_id]["gt_bboxes"], frame_data[vehicle_id]["lidar_pose"]), None, color="g", linewidth=2
            )
            draw_bboxes_2d(
                ax, bbox_sensor_to_map(frame_data[vehicle_id]["pred_bboxes"], frame_data[vehicle_id]["lidar_pose"]), None, color="r", linewidth=2
            )

    error_areas: List[Any] = []
    for vehicle_id in metrics[0]:
        for t in ["spoof", "remove"]:
            error_areas += [x[0] for x in metrics[0][vehicle_id][t]]
    draw_polygons(ax, error_areas, color="y", alpha=0.8, border=False)

    ax.set_aspect("equal", adjustable="box")
    show_or_save(show=show, save=save)


def draw_roc(value: np.ndarray, label: np.ndarray, show: bool = False, save: Optional[str] = None) -> Tuple[float, float, float, float]:
    tpr_data: List[float] = []
    fpr_data: List[float] = []
    roc_auc = 0.0
    best_thres = 0.0
    best_TPR = 0.0
    best_FPR = 0.0
    for thres in np.arange(value.min() - 0.02, value.max() + 0.02, 0.02).tolist():
        TP = np.sum((value > thres) * (label > 0))
        FP = np.sum((value > thres) * (label <= 0))
        P = np.sum(label > 0)
        N = np.sum(label <= 0)
        TPR = TP / P
        FPR = FP / N
        if TPR * (1 - FPR) > roc_auc:
            roc_auc = TPR * (1 - FPR)
            best_thres = thres
            best_TPR = TPR
            best_FPR = FPR
        tpr_data.append(TPR)
        fpr_data.append(FPR)

    plt.plot(fpr_data, tpr_data, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.gca().set_aspect("equal", adjustable="box")
    show_or_save(show=show, save=save)
    return best_TPR, best_FPR, roc_auc, best_thres
