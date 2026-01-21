# -*- coding: utf-8 -*-
"""
Rasterization drawing functions.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import cv2
import numpy.typing as npt
from typing import List

# sub-pixel drawing precision constants
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]

AGENT_COLOR = (255, 255, 255)
ROAD_COLOR = (17, 17, 31)
Lane_COLOR = {"normal": (255, 217, 82), "red": (255, 0, 0), "yellow": (255, 255, 0), "green": (0, 255, 0)}


def cv2_subpixel(coords: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT.

    cv2 calls will use shift to restore original values with higher precision.

    Parameters
    ----------
    coords : NDArray[np.float32]
        XY coordinates as float.

    Returns
    -------
    NDArray[np.int64]
        XY coordinates as int for cv2 shift draw.
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int64)
    return coords


def draw_agent(agent_list: List[npt.NDArray[np.float32]], image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Draw agent mask on image.

    Parameters
    ----------
    agent_list : List[NDArray[np.float32]]
        The agent corner list.
    image : NDArray[np.uint8]
        The image to be drawn.

    Returns
    -------
    NDArray[np.uint8]
        Drawn image.
    """
    for agent_corner in agent_list:
        agent_corner = agent_corner.reshape(-1, 2)
        cv2.fillPoly(image, [agent_corner], AGENT_COLOR, **CV2_SUB_VALUES)
    return image


def draw_road(lane_area_list: List[npt.NDArray[np.float32]], image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Draw poly for road.

    Parameters
    ----------
    lane_area_list : List[NDArray[np.float32]]
        List of lane coordinates.
    image : NDArray[np.uint8]
        Image to be drawn.

    Returns
    -------
    NDArray[np.uint8]
        Drawn image.
    """
    for lane_area in lane_area_list:
        lane_area = lane_area.reshape(-1, 2)
        cv2.fillPoly(image, [lane_area], ROAD_COLOR, **CV2_SUB_VALUES)
    return image


def draw_lane(lane_area_list: List[npt.NDArray[np.float32]], lane_type_list: List[str], image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Draw lanes on image (polylines).

    Parameters
    ----------
    lane_area_list : List[NDArray[np.float32]]
        List of lane coordinates.
    lane_type_list : List[str]
        List of lane types, normal, red, green or yellow.
    image : NDArray[np.uint8]
        Image to be drawn.

    Returns
    -------
    NDArray[np.uint8]
        Drawn image.
    """
    for lane_area, lane_type in zip(lane_area_list, lane_type_list):
        cv2.polylines(image, lane_area, False, Lane_COLOR[lane_type], **CV2_SUB_VALUES)

    return image
