"""
Rasterization utilities for drawing map elements and agents on BEV images.

This module provides functions for rendering various map elements (roads, lanes,
crosswalks, buildings) and vehicle agents onto bird's-eye view (BEV) images using
OpenCV with sub-pixel precision for smooth visualization.
"""

import numpy as np
from numpy.typing import NDArray
import cv2
from typing import List, Dict, Any, Optional

# sub-pixel drawing precision constants
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]
INTERPOLATION_POINTS = 20

AGENT_COLOR = (255, 255, 255)
ROAD_COLOR = (255, 255, 255)
ROAD_COLOR_VIS = (17, 17, 31)
Lane_COLOR = {"normal": (255, 217, 82), "red": (255, 0, 0), "yellow": (255, 255, 0), "green": (0, 255, 0)}
BUILD_COLOR = (70, 70, 70)
Terrain_COLOR = (145, 170, 100)
SideWalk_COLOR = (244, 35, 232)
# color map
OBJ_COLOR_MAP = {"building": BUILD_COLOR, "terrain": Terrain_COLOR, "sidewalk": SideWalk_COLOR}


def cv2_subpixel(coords: NDArray) -> NDArray:
    """
    Cast coordinates to int but keep fractional part.
    
    Multiplies by 2**CV2_SHIFT beforehand. cv2 calls will use shift 
    to restore original values with higher precision.

    Parameters
    ----------
    coords : NDArray
        XY coordinates as float.

    Returns
    -------
    coords : NDArray
        XY coordinates as int for cv2 shift draw.
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords


def draw_agent(agent_list: List[NDArray], image: NDArray) -> NDArray:
    """
    Draw agent mask on image.

    Parameters
    ----------
    agent_list : list
        The agent corner list.

    image : np.ndarray
        The image to be drawn.

    Returns
    -------
    Drawn image.
    """
    for agent_corner in agent_list:
        agent_corner = agent_corner.reshape(-1, 2)
        cv2.fillPoly(image, [agent_corner], AGENT_COLOR, **CV2_SUB_VALUES)
    return image


def draw_road(
    lane_area_list: List[NDArray], 
    image: NDArray, 
    visualize: bool = False
) -> NDArray:
    """
    Draw poly for road.

    Parameters
    ----------
    visualize : bool
        If set to true, the road segment will be black color.

    lane_area_list : list
        List of lane coordinates

    image : np.ndarray
        image to be drawn

    Returns
    -------
    drawed image.
    """
    color = ROAD_COLOR if not visualize else ROAD_COLOR_VIS

    for lane_area in lane_area_list:
        lane_area = lane_area.reshape(-1, 2)
        cv2.fillPoly(image, [lane_area], color, **CV2_SUB_VALUES)
    return image


def road_exclude(static_road: NDArray) -> NDArray:
    """
    Exclude the road segment that is not connected to the ego vehicle
    position.

    Parameters
    ----------
    static_road : np.ndarray
        The static bev map with road segment.

    Returns
    -------
    The road without unrelated road.
    """
    binary_bev = cv2.cvtColor(static_road, cv2.COLOR_BGR2GRAY)
    _, label, stats, _ = cv2.connectedComponentsWithStats(binary_bev)

    ego_label = label[static_road.shape[0] // 2, static_road.shape[1] // 2]
    static_road[label != ego_label] = 0

    return static_road


def draw_lane(
    lane_area_list: List[NDArray],
    lane_type_list: List[str],
    image: NDArray,
    intersection_list: Optional[List[bool]] = None,
    vis: bool = True
) -> NDArray:
    """
    Draw lanes on image (polylines).

    Parameters
    ----------
    intersection_list : list
    lane_area_list : list
        List of lane coordinates

    lane_type_list : list
        List of lane types, normal, red, green or yellow.

    image : np.ndarray
        image to be drawn

    vis : bool
        Whether to visualize

    Returns
    -------
    drawed image.
    """
    if intersection_list is None:
        intersection_list = [False] * len(lane_area_list)

    for lane_area, lane_type, inter_flag in zip(lane_area_list, lane_type_list, intersection_list):
        if inter_flag:
            continue
        cv2.polylines(image, lane_area, False, Lane_COLOR[lane_type] if vis else (255, 255, 255), **CV2_SUB_VALUES)

    return image


def draw_crosswalks(lane_area_list: List[NDArray], image: NDArray) -> NDArray:
    """
    Draw lanes on image (polylines).

    Parameters
    ----------
    lane_area_list : list
        List of cross coordinates

    image : np.ndarray
        image to be drawn

    vis : bool
        Whether to visualize

    Returns
    -------
    drawed image.
    """
    for lane_area in lane_area_list:
        up_line = lane_area[0]
        bottom_line = lane_area[1]

        cv2.line(image, (up_line[0, 0], up_line[0, 1]), (up_line[-1, 0], up_line[-1, 1]), (255, 255, 255), 2, **CV2_SUB_VALUES)
        cv2.line(image, (bottom_line[0, 0], bottom_line[0, 1]), (bottom_line[-1, 0], bottom_line[-1, 1]), (255, 255, 255), 2, **CV2_SUB_VALUES)
        cv2.line(image, (up_line[0, 0], up_line[0, 1]), (bottom_line[-1, 0], bottom_line[-1, 1]), (255, 255, 255), 2, **CV2_SUB_VALUES)
        cv2.line(image, (up_line[-1, 0], up_line[-1, 1]), (bottom_line[0, 0], bottom_line[0, 1]), (255, 255, 255), 2, **CV2_SUB_VALUES)
    return image


def draw_city_objects(
    city_obj_info: Dict[str, Dict[str, Any]], 
    image: NDArray
) -> NDArray:

    """
    Draw static objects other than lane, road, crosswalks on image.

    Parameters
    ----------
    city_obj_info : dict
    image : np.ndarray

    Returns
    -------
    Drew image.
    """
    for obj_category, obj_content in city_obj_info.items():
        for _, obj in obj_content.items():
            obj_corner = obj["corner_area"].reshape(-1, 2)
            cv2.fillPoly(image, [obj_corner], OBJ_COLOR_MAP[obj_category], **CV2_SUB_VALUES)
    return image
