"""
Visualization utilities for 3D object detection and point cloud data.

This module provides functions for visualizing point clouds, bounding boxes,
and predictions using Open3D and Matplotlib.
"""

import time
import torch
from typing import List, Tuple, Any, Union

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from opencood.utils import box_utils
from opencood.utils import common_utils

VIRIDIS = np.array(cm.get_cmap("plasma").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def bbx2linset(bbx_corner: Union[npt.NDArray, Any], order: str = "hwl", color: Tuple[float, float, float] = (0, 1, 0)) -> List[o3d.geometry.LineSet]:
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor or np.ndarray
        Bounding box corners with shape
    order : str, optional
        The order of the bounding box
    color : tuple of float, optional
        The bounding box color as RGB

    Returns
    -------
    list of o3d.geometry.LineSet
        The list containing linesets.
    """

    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = -bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(
    bbx_corner: Union[npt.NDArray, Any], order: str = "hwl", color: Tuple[float, float, float] = (0, 0, 1)
) -> List[o3d.geometry.OrientedBoundingBox]:
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor or np.ndarray
        Bounding box corners with shape (n, 8, 3).
    order : str, optional
        The order of the bounding box if shape is (n, 7). Default is "hwl".
    color : tuple of float, optional
        The bounding box color as RGB. Default is (0, 0, 1).

    Returns
    -------
    list of o3d.geometry.OrientedBoundingBox
        The list containing all oriented bounding boxes.
    """

    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = -bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def linset_assign_list(
    vis: o3d.visualization.Visualizer,
    lineset_list1: List[o3d.geometry.LineSet],
    lineset_list2: List[o3d.geometry.LineSet],
    update_mode: str = "update",
) -> None:
    """
    Associate two lists of lineset.

    Parameters
    ----------
    vis : o3d.visualization.Visualizer
        Open3D visualizer instance.
    lineset_list1 : list of o3d.geometry.LineSet
        First list of linesets.
    lineset_list2 : list of o3d.geometry.LineSet
        Second list of linesets.
    update_mode : str, optional
        Mode for geometry update, either "add" or "update". Default is "update".
    """
    for j in range(len(lineset_list1)):
        index = j if j < len(lineset_list2) else -1
        if len(lineset_list2):
            lineset_list1[j] = lineset_assign(lineset_list1[j], lineset_list2[index])
        if update_mode == "add":
            vis.add_geometry(lineset_list1[j])
        else:
            vis.update_geometry(lineset_list1[j])


def lineset_assign(lineset1: o3d.geometry.LineSet, lineset2: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : o3d.geometry.LineSet
        Target lineset to be updated.
    lineset2 : o3d.geometry.LineSet
        Source lineset with attributes to copy.

    Returns
    -------
    o3d.geometry.LineSet
        The lineset1 object with lineset2's attributes.
    """
    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors

    return lineset1


def color_encoding(intensity: npt.NDArray[np.floating], mode: str = "intensity") -> npt.NDArray[np.floating]:
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity with shape (n,).
    mode : str, optional
        The color rendering mode. Options are "intensity", "z-value", or "constant".
        Default is "intensity".

    Returns
    -------
    np.ndarray
        Encoded Lidar color with shape (n, 3).
    """
    assert mode in ["intensity", "z-value", "constant"]

    if mode == "intensity":
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2]),
        ]

    elif mode == "z-value":
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]

    elif mode == "constant":
        # regard all point cloud the same color
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255

    return int_color


def visualize_single_sample_output_gt(
    pred_tensor: Union[npt.NDArray, Any],
    gt_tensor: Union[npt.NDArray, Any],
    pcd: Union[npt.NDArray, Any],
    show_vis: bool = True,
    save_path: str = "",
    mode: str = "constant",
) -> None:
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor or np.ndarray
        Prediction bounding boxes with shape (N, 8, 3).
    gt_tensor : torch.Tensor or np.ndarray
        Groundtruth bounding boxes with shape (N, 8, 3).
    pcd : torch.Tensor or np.ndarray
        PointCloud with shape (N, 4).
    show_vis : bool, optional
        Whether to show visualization. Default is True.
    save_path : str, optional
        Save the visualization results to given path. Default is "".
    mode : str, optional
        Color rendering mode. Default is "constant".
    """

    def custom_draw_geometry(pcd: o3d.geometry.PointCloud, pred: List[o3d.geometry.LineSet], gt: List[o3d.geometry.LineSet]) -> None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # noqa: DC05
        opt.point_size = 1.0  # noqa: DC05

        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    if len(pcd.shape) == 3:
        pcd = pcd[0]
    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    origin_lidar_intcolor = color_encoding(origin_lidar[:, -1] if mode == "intensity" else origin_lidar[:, 2], mode=mode)
    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))

    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)


def visualize_single_sample_output_bev(
    pred_box: Union[npt.NDArray, Any],
    gt_box: Union[npt.NDArray, Any],
    pcd: Union[npt.NDArray, Any],
    dataset: Any,
    show_vis: bool = True,
    save_path: str = "",
) -> None:
    """
    Visualize the prediction, groundtruth with point cloud together in a bev format.

    Parameters
    ----------
    pred_box : torch.Tensor or np.ndarray
        Prediction bounding boxes with shape (N, 4, 2).
    gt_box : torch.Tensor or np.ndarray
        Groundtruth bounding boxes with shape (N, 4, 2).
    pcd : torch.Tensor or np.ndarray
        PointCloud with shape (N, 4).
    dataset : object
        Dataset object containing preprocessing parameters.
    show_vis : bool, optional
        Whether to show visualization. Default is True.
    save_path : str, optional
        Save the visualization results to given path. Default is "".
    """
    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pred_box is not None and not isinstance(pred_box, np.ndarray):
        pred_box = common_utils.torch_tensor_to_numpy(pred_box)
    if gt_box is not None and not isinstance(gt_box, np.ndarray):
        gt_box = common_utils.torch_tensor_to_numpy(gt_box)

    ratio = dataset.params["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, _ = dataset.params["preprocess"]["cav_lidar_range"]
    bev_origin = np.array([L1, W1]).reshape(1, -1)
    # (img_row, img_col)
    bev_map = dataset.project_points_to_bev_map(pcd, ratio)
    # (img_row, img_col, 3)
    bev_map = np.repeat(bev_map[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    bev_map = bev_map * 255

    if pred_box is not None:
        num_bbx = pred_box.shape[0]
        for i in range(num_bbx):
            bbx = pred_box[i]

            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (0, 0, 255), 1)

    if gt_box is not None and len(gt_box):
        for i in range(gt_box.shape[0]):
            bbx = gt_box[i][:4, :2]
            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (255, 0, 0), 1)

    if show_vis:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.show()
    if save_path:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.savefig(save_path)


def visualize_single_sample_dataloader(
    batch_data: dict,
    o3d_pcd: o3d.geometry.PointCloud,
    order: str,
    key: str = "origin_lidar",
    visualize: bool = False,
    save_path: str = "",
    oabb: bool = False,
    mode: str = "constant",
) -> Tuple[o3d.geometry.PointCloud, List]:
    """
    Visualize a single frame of a single CAV for validation of data pipeline.

    Parameters
    ----------
    batch_data : dict
        The dictionary that contains current timestamp's data.
    o3d_pcd : o3d.geometry.PointCloud
        Open3D PointCloud object.
    order : str
        The bounding box order.
    key : str, optional
        Key for lidar data, "origin_lidar" for late fusion and "stacked_lidar" for early fusion.
        Default is "origin_lidar".
    visualize : bool, optional
        Whether to visualize the sample. Default is False.
    save_path : str, optional
        If set, save the visualization image to the path. Default is "".
    oabb : bool, optional
        If oriented bounding box is used. Default is False.
    mode : str, optional
        Color rendering mode. Default is "constant".

    Returns
    -------
    tuple
        A tuple containing (o3d_pcd, aabbs).
    """
    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = color_encoding(origin_lidar[:, -1] if mode == "intensity" else origin_lidar[:, 2], mode=mode)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    object_bbx_center = batch_data["object_bbx_center"]
    object_bbx_mask = batch_data["object_bbx_mask"]
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    aabbs = bbx2linset(object_bbx_center, order) if not oabb else bbx2oabb(object_bbx_center, order)
    visualize_elements = [o3d_pcd] + aabbs
    if visualize:
        o3d.visualization.draw_geometries(visualize_elements)

    if save_path:
        save_o3d_visualization(visualize_elements, save_path)

    return o3d_pcd, aabbs


def visualize_inference_sample_dataloader(
    pred_box_tensor: Union[npt.NDArray, Any],
    gt_box_tensor: Union[npt.NDArray, Any],
    origin_lidar: Union[npt.NDArray, Any],
    o3d_pcd: o3d.geometry.PointCloud,
    mode: str = "constant",
) -> Tuple[o3d.geometry.PointCloud, List[o3d.geometry.LineSet], List[o3d.geometry.LineSet]]:
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor or np.ndarray
        Prediction bounding boxes with shape (N, 8, 3).
    gt_box_tensor : torch.Tensor or np.ndarray
        Groundtruth bounding boxes with shape (N, 8, 3).
    origin_lidar : torch.Tensor or np.ndarray
        PointCloud with shape (N, 4).
    o3d_pcd : o3d.geometry.PointCloud
        Open3D PointCloud used to visualize the point cloud.
    mode : str, optional
        Lidar point rendering mode. Default is "constant".

    Returns
    -------
    tuple
        A tuple containing (o3d_pcd, pred_o3d_box, gt_o3d_box).
    """
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    # this is for 2-stage origin lidar, it has different format
    if origin_lidar.shape[1] > 4:
        origin_lidar = origin_lidar[:, 1:]

    origin_lidar_intcolor = color_encoding(origin_lidar[:, -1] if mode == "intensity" else origin_lidar[:, 2], mode=mode)

    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    gt_o3d_box = bbx2linset(gt_box_tensor, order="hwl", color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))

    return o3d_pcd, pred_o3d_box, gt_o3d_box


def visualize_sequence_dataloader(dataloader: torch.utils.data.DataLoader, order: str, color_mode: str = "constant") -> None:
    """
    Visualize the batch data in animation.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Pytorch dataloader.
    order : str
        Bounding box order (N, 7).
    color_mode : str, optional
        Color rendering mode. Default is "constant".
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().background_color = [0.05, 0.05, 0.05]  # noqa: DC05
    vis.get_render_option().point_size = 1.0  # noqa: DC05
    vis.get_render_option().show_coordinate_frame = True  # noqa: DC05

    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()
    # used to visualize object bounding box, maximum 50
    vis_aabbs = []
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())

    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)
            pcd, aabbs = visualize_single_sample_dataloader(sample_batched["ego"], vis_pcd, order, mode=color_mode)
            if i_batch == 0:
                vis.add_geometry(pcd)
                for i in range(len(vis_aabbs)):
                    index = i if i < len(aabbs) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                    vis.add_geometry(vis_aabbs[i])

            for i in range(len(vis_aabbs)):
                index = i if i < len(aabbs) else -1
                vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                vis.update_geometry(vis_aabbs[i])

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    vis.destroy_window()


def save_o3d_visualization(element: List, save_path: str) -> None:
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.
    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()
