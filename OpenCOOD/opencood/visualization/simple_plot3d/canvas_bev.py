"""
Simple BEV visualization for 3D points & boxes.

Written by Jinhyung Park

This module provides canvas classes for bird's eye view (BEV) visualization
of 3D point clouds and bounding boxes in different coordinate systems.
"""

from typing import Tuple, Optional, Union, List

import numpy as np
import numpy.typing as npt
import cv2 
import matplotlib 


class Canvas_BEV_heading_right(object):
    """
    BEV canvas optimized for forward-facing view (heading right).

    Similar to Canvas_BEV but with landscape orientation and adjusted
    coordinate transformation for vehicle-centric visualization.

    Parameters
    ----------
    canvas_shape : tuple of int, optional
        Canvas dimensions (height, width) in pixels.
    canvas_x_range : tuple of float, optional
        World x-axis range (min, max) in meters.
    canvas_y_range : tuple of float, optional
        World y-axis range (min, max) in meters.
    canvas_bg_color : tuple of int, optional
        Background RGB color (0-255).
    left_hand : bool, optional
        If True, uses left-hand coordinate system. Default is True.

    Attributes
    ----------
    canvas : np.ndarray
        Current canvas image with shape (height, width, 3) in BGR format.
    """

    def __init__(self, canvas_shape=(800, 2800), canvas_x_range=(-140, 140), canvas_y_range=(-40, 40), canvas_bg_color=(0, 0, 0), left_hand=True):
        # Sanity check ratios
        if (canvas_shape[1] / canvas_shape[0]) != ((canvas_x_range[0] - canvas_x_range[1]) / (canvas_y_range[0] - canvas_y_range[1])):
            print("Not an error, but the x & y ranges are not proportional to canvas height & width.")

        self.canvas_shape = canvas_shape
        self.canvas_x_range = canvas_x_range
        self.canvas_y_range = canvas_y_range
        self.canvas_bg_color = canvas_bg_color
        self.left_hand = left_hand

        self.clear_canvas()

    def clear_canvas(self):
        self.canvas = np.zeros((*self.canvas_shape, 3), dtype=np.uint8)
        self.canvas[..., :] = self.canvas_bg_color

    def get_canvas_coords(self, xy: npt.NDArray[np.floating]) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
        """
        Transform world coordinates to canvas coordinates.

        Parameters
        ----------
        xy : npt.NDArray[np.floating]
            Array of coordinates with shape (N, 2+). Additional columns beyond
            the first two are ignored.

        Returns
        -------
        canvas_xy : npt.NDArray[np.int32]
            Array with shape (N, 2) of xy scaled into canvas coordinates.
            Invalid locations are clipped into range. "x" is dim0, "y" is dim1 of canvas.
        valid_mask : npt.NDArray[np.bool_]
            Boolean mask with shape (N,) indicating which canvas_xy points fit into canvas.
        """
        xy = np.copy(xy)  # prevent in-place modifications

        x = xy[:, 0]
        y = xy[:, 1]

        if not self.left_hand:
            y = -y

        # Get valid mask
        valid_mask = (x > self.canvas_x_range[0]) & (x < self.canvas_x_range[1]) & (y > self.canvas_y_range[0]) & (y < self.canvas_y_range[1])

        # Rescale points
        # They are exactly lidar point coordinate
        x = (x - self.canvas_x_range[0]) / (self.canvas_x_range[1] - self.canvas_x_range[0])
        x = x * self.canvas_shape[1]
        x = np.clip(np.around(x), 0, self.canvas_shape[1] - 1).astype(np.int32)  # [0,2800-1]

        y = (y - self.canvas_y_range[0]) / (self.canvas_y_range[1] - self.canvas_y_range[0])
        y = y * self.canvas_shape[0]
        y = np.clip(np.around(y), 0, self.canvas_shape[0] - 1).astype(np.int32)  # [0,800-1]

        # x and y are exactly image coordinate
        # ------------> x
        # |
        # |
        # |
        # y

        canvas_xy = np.stack([x, y], axis=1)

        return canvas_xy, valid_mask

    def draw_canvas_points(
        self,
        canvas_xy: npt.NDArray[np.int32],
        radius: int = -1,
        colors: Optional[Union[Tuple[int, int, int], npt.NDArray[np.uint8], str]] = None,
        colors_operand: Optional[npt.NDArray[np.floating]] = None,
    ) -> None:
        """
        Draw points onto the canvas.

        Parameters
        ----------
        canvas_xy : npt.NDArray[np.int32]
            Array with shape (N, 2) of valid canvas coordinates.
        radius : int, optional
            Point radius. -1 means each point is a single pixel, otherwise
            points are circles with given radius. Default is -1.
        colors : tuple or npt.NDArray[np.uint8] or str or None, optional
            Color specification:
            - None: colors all points white
            - Tuple: RGB (0 ~ 255), single color for all points
            - ndarray: (N, 3) array of RGB values for each point
            - String: matplotlib cmap name like "Spectral"
            Default is None.
        colors_operand : npt.NDArray[np.floating] or None, optional
            Array with shape (N,) of values corresponding to canvas_xy,
            used only if colors is a cmap. Default is None.
        """
        if len(canvas_xy) == 0:
            return

        if colors is None:
            colors = np.full((len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., :] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        elif isinstance(colors, str):
            colors = matplotlib.cm.get_cmap(colors)
            if colors_operand is None:
                # Get distances from (0, 0) (albeit potentially clipped)
                origin_center = self.get_canvas_coords(np.zeros((1, 2)))[0][0]
                colors_operand = np.sqrt(((canvas_xy - origin_center) ** 2).sum(axis=1))

            # Normalize 0 ~ 1 for cmap
            colors_operand = colors_operand - colors_operand.min()
            colors_operand = colors_operand / colors_operand.max()

            # Get cmap colors - note that cmap returns (*input_shape, 4), with
            # colors scaled 0 ~ 1
            colors = (colors(colors_operand)[:, :3] * 255).astype(np.uint8)
        else:
            raise Exception("colors type {} was not an expected type".format(type(colors)))

        # Here the order is different from Canvas_BEV
        if radius == -1:
            self.canvas[canvas_xy[:, 1], canvas_xy[:, 0], :] = colors
        else:
            for color, (x, y) in zip(colors.tolist(), canvas_xy.tolist()):
                self.canvas = cv2.circle(self.canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    def draw_boxes(
        self,
        boxes: npt.NDArray[np.floating],
        colors: Optional[Union[Tuple[int, int, int], npt.NDArray[np.uint8]]] = None,
        texts: Optional[List[str]] = None,
        box_line_thickness: int = 2,
        box_text_size: float = 0.5,
        text_corner: int = 0,
    ) -> None:
        """
        Draw bounding boxes onto the canvas.

        Parameters
        ----------
        boxes : npt.NDArray[np.floating]
            Array with shape [N, 8, 3] of 3D box corners.
        colors : tuple or npt.NDArray[np.uint8] or None, optional
            Color specification:
            - None: colors all boxes white
            - Tuple: RGB (0 ~ 255), single color for all boxes
            - ndarray: (N, 3) array of RGB values for each box
            Default is None.
        texts : list of str or None, optional
            Length N list of text to write next to boxes. Default is None.
        box_line_thickness : int, optional
            cv2 line/text thickness. Default is 2.
        box_text_size : float, optional
            cv2 putText size. Default is 0.5.
        text_corner : int, optional
            Corner index (0 ~ 3) of 3D box to write text at. Default is 0.
        """
        # Setup colors
        if colors is None:
            colors = np.full((len(boxes), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(boxes), 3), dtype=np.uint8)
            colors_tmp[..., : len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(boxes)
            colors = colors.astype(np.uint8)
        else:
            raise Exception("colors type {} was not an expected type".format(type(colors)))

        boxes = np.copy(boxes)  # prevent in-place modifications

        # Translate BEV 4 corners, [N, 4, 2]
        #     4 -------- 5
        #    /|         /|
        #   7 -------- 6 .
        #   | |        | |
        #   . 0 -------- 1
        #   |/         |/
        #   3 -------- 2
        bev_corners = boxes[:, :4, :2]

        ## Transform BEV 4 corners to canvas coords
        bev_corners_canvas, valid_mask = self.get_canvas_coords(bev_corners.reshape(-1, 2))  # [N, 2]
        bev_corners_canvas = bev_corners_canvas.reshape(*bev_corners.shape)  # [N, 4, 2]
        valid_mask = valid_mask.reshape(*bev_corners.shape[:-1])

        # At least 1 corner in canvas to draw.
        valid_mask = valid_mask.sum(axis=1) > 0
        bev_corners_canvas = bev_corners_canvas[valid_mask]
        if texts is not None:
            texts = np.array(texts)[valid_mask]

        ## Draw onto canvas
        # Draw the outer boundaries
        idx_draw_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, (color, curr_box_corners) in enumerate(zip(colors.tolist(), bev_corners_canvas)):
            curr_box_corners = curr_box_corners.astype(np.int32)
            for start, end in idx_draw_pairs:
                # Notice Difference Here
                self.canvas = cv2.line(
                    self.canvas,
                    tuple(curr_box_corners[start].tolist()),
                    tuple(curr_box_corners[end].tolist()),
                    color=color,
                    thickness=box_line_thickness,
                )
            if texts is not None:
                self.canvas = cv2.putText(
                    self.canvas,
                    str(texts[i]),
                    tuple(curr_box_corners[text_corner].tolist()),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    box_text_size,
                    color=color,
                    thickness=box_line_thickness,
                )
