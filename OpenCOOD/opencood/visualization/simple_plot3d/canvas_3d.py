"""
Written by Jinhyung Park

Simple 3D visualization for 3D points & boxes. Intended as a simple, hackable
alternative to mayavi for certain point cloud tasks.
"""

import numpy as np
import numpy.typing as npt
import cv2
import matplotlib

from typing import Tuple, Optional, Union, List


class Canvas_3D(object):
    """
    3D perspective canvas for visualizing point clouds and bounding boxes.

    Projects 3D points and boxes onto 2D canvas using virtual camera with
    configurable extrinsic and intrinsic parameters.

    Parameters
    ----------
    canvas_shape : tuple of int, optional
        Canvas dimensions (height, width) in pixels.
    camera_center_coords : tuple of float, optional
        Camera position in 3D world coordinates (x, y, z).
    camera_focus_coords : tuple of float, optional
        Point in 3D space the camera looks at (x, y, z).
        Absolute coordinates, not relative to camera center.
    focal_length : int or None, optional
        Camera focal length in pixels:
            - None: Auto-set to max(height, width) // 2
            - int: Explicit value
    canvas_bg_color : tuple of int, optional
        Background RGB color (0-255). Default is (0, 0, 0) (black).
    left_hand : bool, optional
        If True, uses left-hand coordinate system (negates y-axis).

    Attributes
    ----------
    canvas : np.ndarray
        Current canvas image with shape (height, width, 3) in BGR format.
    ext_matrix : np.ndarray
        Camera extrinsic matrix (4, 4) for world-to-camera transformation.
    int_matrix : np.ndarray
        Camera intrinsic matrix (3, 4) for camera-to-image projection.

    """

    def __init__(
        self,
        canvas_shape: Tuple[int, int] = (500, 1000),
        camera_center_coords: Tuple[float, float, float] = (-15, 0, 10),
        camera_focus_coords: Tuple[float, float, float] = (-15 + 0.9396926, 0, 10 - 0.44202014),
        #  camera_center_coords=(-25, 0, 20),
        #  camera_focus_coords=(-25 + 0.9396926, 0, 20 - 0.64202014),
        focal_length: Optional[int] = None,
        canvas_bg_color: Tuple[int, int, int] = (0, 0, 0),
        left_hand: bool = True,
    ):
        self.canvas_shape = canvas_shape
        self.H, self.W = self.canvas_shape
        self.canvas_bg_color = canvas_bg_color
        self.left_hand = left_hand
        if left_hand:
            camera_center_coords = list(camera_center_coords)
            camera_center_coords[1] = -camera_center_coords[1]
            camera_center_coords = tuple(camera_center_coords)

            camera_focus_coords = list(camera_focus_coords)
            camera_focus_coords[1] = -camera_focus_coords[1]
            camera_focus_coords = tuple(camera_focus_coords)

        self.camera_center_coords = camera_center_coords
        self.camera_focus_coords = camera_focus_coords

        if focal_length is None:
            self.focal_length = max(self.H, self.W) // 2
        else:
            self.focal_length = focal_length

        # Setup extrinsics and intrinsics of this virtual camera.
        self.ext_matrix = self.get_extrinsic_matrix(self.camera_center_coords, self.camera_focus_coords)
        self.int_matrix = np.array(
            [
                [self.focal_length, 0, self.W // 2, 0],
                [0, self.focal_length, self.H // 2, 0],
                [0, 0, 1, 0],
            ]
        )

        self.clear_canvas()

    def get_canvas(self):
        """
        Get the current canvas image.

        Returns
        -------
        canvas : np.ndarray
        """
        return self.canvas

    def clear_canvas(self):
        """
        Clear canvas and reset to background color.

        Creates a new blank canvas filled with the background color.
        """
        self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.canvas[..., :] = self.canvas_bg_color

    def get_canvas_coords(
        self,
        xyz: npt.NDArray[np.floating],
        depth_min: float = 0.1,
        return_depth: bool = False,
    ) -> Union[
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_]],
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_], npt.NDArray[np.floating]],
    ]:
        """
        Project 3D points onto 2D canvas using perspective projection.

        Parameters
        ----------
        xyz : np.ndarray
            3D coordinates with shape (N, 3+). Additional columns beyond
            the first three are ignored.
        depth_min : float, optional
            Minimum depth threshold. Points with depth <= this value are
            marked as invalid. Default is 0.1.
        return_depth : bool, optional
            If True, also returns depth values. Default is False.

        Returns
        -------
        canvas_xy : np.ndarray
            Projected canvas coordinates with shape (N, 2).
            Format: [x_pixel, y_pixel] where x is height, y is width.
        valid_mask : npt.ndarray
            Boolean mask with shape (N,) indicating visible points
            (positive depth and within canvas bounds).
        depth : npt.ndarray, optional
            Depth values with shape (N,). Only returned if return_depth=True.
        """
        if self.left_hand:
            xyz[:, 1] = -xyz[:, 1]

        xyz = xyz[:, :3]
        xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        img_pts = (self.int_matrix @ self.ext_matrix @ xyz_hom.T).T

        depth = img_pts[:, 2]
        xy = img_pts[:, :2] / depth[:, None]
        xy_int = xy.round().astype(np.int32)

        # Flip X and Y so "x" is dim0, "y" is dim1 of canvas
        xy_int = xy_int[:, ::-1]

        valid_mask = (depth > depth_min) & (xy_int[:, 0] >= 0) & (xy_int[:, 0] < self.H) & (xy_int[:, 1] >= 0) & (xy_int[:, 1] < self.W)

        if return_depth:
            return xy_int, valid_mask, depth
        else:
            return xy_int, valid_mask

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
        canvas_xy : np.ndarray
            Valid canvas coordinates with shape (N, 2).
            Format: [x_pixel, y_pixel].
        radius : int, optional
            Point rendering radius:
                - -1: Single pixel per point
                - >0: Circle with specified radius
            Default is -1.
        colors : tuple or npt.NDArray[np.uint8] or str or None, optional
            Point color specification:
                - None: All points white (255, 255, 255)
                - Tuple: Single RGB color (0-255) for all points
                - ndarray: Per-point RGB colors with shape (N, 3)
                - str: Matplotlib colormap name (requires colors_operand)
            Default is None.
        colors_operand : npt.NDArray[np.floating] or None, optional
            Values with shape (N,) for colormap mapping. Required when
            colors is a string. Default is None.

        Raises
        ------
        AssertionError
            If colors is a string but colors_operand is None.
        """
        if len(canvas_xy) == 0:
            return

        if colors is None:
            colors = np.full((len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., : len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        elif isinstance(colors, str):
            assert colors_operand is not None
            colors = matplotlib.cm.get_cmap(colors)

            # Normalize 0 ~ 1 for cmap
            colors_operand = colors_operand - colors_operand.min()
            colors_operand = colors_operand / colors_operand.max()

            # Get cmap colors - note that cmap returns (*input_shape, 4), with
            # colors scaled 0 ~ 1
            colors = (colors(colors_operand)[:, :3] * 255).astype(np.uint8)
        else:
            raise Exception("colors type {} was not an expected type".format(type(colors)))

        if radius == -1:
            self.canvas[canvas_xy[:, 0], canvas_xy[:, 1], :] = colors
        else:
            for color, (x, y) in zip(colors.tolist(), canvas_xy.tolist()):
                self.canvas = cv2.circle(self.canvas, (y, x), radius, color, -1, lineType=cv2.LINE_AA)

    def draw_lines(
        self,
        canvas_xy: npt.NDArray[np.int32],
        start_xyz: npt.NDArray[np.floating],
        end_xyz: npt.NDArray[np.floating],
        colors: Optional[Union[Tuple[int, int, int], npt.NDArray[np.uint8]]] = (
            255,
            255,
            255,
        ),
        thickness: int = 1,
    ) -> None:
        """
        Draw lines between 3D points on the canvas.

        Parameters
        ----------
        canvas_xy : np.ndarray
            Valid canvas coordinates with shape (N, 2).
        start_xyz : np.ndarray
            3D starting points with shape (N, 3).
        end_xyz : np.ndarray
            3D ending points with shape (N, 3). Same length as start_xyz.
        colors : tuple or np.ndarray or None, optional
            Line color specification:
                - None: All lines white
                - Tuple: Single RGB color (0-255) for all lines
                - ndarray: Per-line RGB colors with shape (N, 3)
        thickness : int, optional
            Line thickness in pixels. D
        """
        if colors is None:
            colors = np.full((len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., : len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        else:
            raise Exception("colors type {} was not an expected type".format(type(colors)))

        start_pts_xy, start_pts_valid_mask, start_pts_d = self.get_canvas_coords(start_xyz, True)
        end_pts_xy, end_pts_valid_mask, end_pts_d = self.get_canvas_coords(end_xyz, True)

        for idx, (color, start_pt_xy, end_pt_xy) in enumerate(zip(colors.tolist(), start_pts_xy.tolist(), end_pts_xy.tolist())):
            if start_pts_valid_mask[idx] and end_pts_valid_mask[idx]:
                self.canvas = cv2.line(
                    self.canvas, tuple(start_pt_xy[::-1]), tuple(end_pt_xy[::-1]), color=color, thickness=thickness, lineType=cv2.LINE_AA
                )

    def draw_boxes(
        self,
        boxes: npt.NDArray[np.floating],
        colors: Optional[Union[Tuple[int, int, int], npt.NDArray[np.uint8]]] = None,
        texts: Optional[List[str]] = None,
        depth_min: float = 0.1,
        draw_incomplete_boxes: bool = False,
        box_line_thickness: int = 2,
        box_text_size: float = 0.5,
        text_corner: int = 1,
    ) -> None:
        """
        Draw 3D bounding boxes on the canvas.

        Parameters
        ----------
        boxes : np.ndarray
            3D bounding box corners with shape (N, 8, 3).
            Corner ordering:
                4 -------- 5
               /|         /|
              7 -------- 6 .
              | |        | |
              . 0 -------- 1
              |/         |/
              3 -------- 2
        colors : tuple or np.ndarray or None, optional
            Box color specification.
        texts : list of str or None, optional
            Text labels for each box (length N).
        depth_min : float, optional
            Minimum depth threshold for corner visibility.
        draw_incomplete_boxes : bool, optional
            If False, only draws boxes with all 8 corners visible.
            If True, draws partial boxes.
        box_line_thickness : int, optional
            Line thickness for box edges.
        box_text_size : float, optional
            Text size scale factor.
        text_corner : int, optional
            Corner index (0-7) where text is placed.
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

        corners = boxes  # N x 8 x 3

        # Now we have corners. Need them on the canvas 2D space.
        corners_xy, valid_mask = self.get_canvas_coords(corners.reshape(-1, 3), depth_min=depth_min)
        corners_xy = corners_xy.reshape(-1, 8, 2)
        valid_mask = valid_mask.reshape(-1, 8)

        # Now draw them with lines in correct places
        for i, (color, curr_corners_xy, curr_valid_mask) in enumerate(zip(colors.tolist(), corners_xy.tolist(), valid_mask.tolist())):
            if not draw_incomplete_boxes and sum(curr_valid_mask) != 8:
                # Some corner is invalid, don't draw the box at all.
                continue

            for start, end in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]:
                if not (curr_valid_mask[start] and curr_valid_mask[end]):
                    continue  # start or end is not valid

                self.canvas = cv2.line(
                    self.canvas,
                    (curr_corners_xy[start][1], curr_corners_xy[start][0]),
                    (curr_corners_xy[end][1], curr_corners_xy[end][0]),
                    color=color,
                    thickness=box_line_thickness,
                    lineType=cv2.LINE_AA,
                )

            # If even a single line was drawn, add text as well.
            if sum(curr_valid_mask) > 0:
                if texts is not None:
                    self.canvas = cv2.putText(
                        self.canvas,
                        str(texts[i]),
                        (curr_corners_xy[text_corner][1], curr_corners_xy[text_corner][0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        box_text_size,
                        color,
                        thickness=box_line_thickness,
                    )

    @staticmethod
    def cart2sph(xyz: npt.NDArray[np.floating]) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Parameters
        ----------
        xyz : np.ndarray
            Cartesian coordinates with shape (N, 3).

        Returns
        -------
        az : np.ndarray
            Azimuth angles in radians with shape (N,).
            Measured from +x axis, counter-clockwise.
        el : np.ndarray
            Elevation angles in radians with shape (N,).
            Measured from xy-plane.
        depth : np.ndarray
            Radial distances with shape (N,).
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        depth = np.linalg.norm(xyz, 2, axis=1)
        az = -np.arctan2(y, x)
        el = np.arcsin(z / depth)
        return az, el, depth

    @staticmethod
    def get_extrinsic_matrix(
        camera_center_coords: Tuple[float, float, float],
        camera_focus_coords: Tuple[float, float, float],
    ) -> npt.NDArray[np.floating]:
        """
        Compute camera extrinsic matrix from position and look-at point.

        Parameters
        ----------
        camera_center_coords : tuple of float
            Camera position (x, y, z) in 3D world coordinates.
        camera_focus_coords : tuple of float
            Look-at point (x, y, z) in 3D world coordinates.

        Returns
        -------
        ext_matrix : np.ndarray
            Extrinsic matrix with shape (4, 4) for world-to-camera
            transformation in homogeneous coordinates.
        """
        center_x, center_y, center_z = camera_center_coords
        focus_x, focus_y, focus_z = camera_focus_coords
        az, el, depth = Canvas_3D.cart2sph(np.array([[focus_x - center_x, focus_y - center_y, focus_z - center_z]]))
        az = float(az)
        el = float(el)
        depth = float(depth)

        ### First, construct extrinsics
        ## Rotation matrix

        z_rot = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])

        # el is rotation around y axis.
        y_rot = np.array(
            [
                [np.cos(-el), 0, -np.sin(-el)],
                [0, 1, 0],
                [np.sin(-el), 0, np.cos(-el)],
            ]
        )

        ## Now, how the z_rot and y_rot work (spherical coordiantes), is it
        ## computes rotations starting from the positive x axis and rotates
        ## positive x axis to the desired direction. The desired direction is
        ## the "looking direction" of the camera, which should actually be the
        ## z-axis. So should convert the points so that the x axis is the new z
        ## axis, and after the transformations.
        ## Why x -> z for points? If we think about rotating the camera, z
        ## should become x, so reverse when moving points.
        last_rot = np.array(
            [
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],  # x -> z
            ]
        )

        # Put them together. Order matters. Make it hom.
        rot_matrix = np.eye(4, dtype=np.float32)
        rot_matrix[:3, :3] = last_rot @ y_rot @ z_rot

        ## Translation matrix
        trans_matrix = np.array(
            [
                [1, 0, 0, -center_x],
                [0, 1, 0, -center_y],
                [0, 0, 1, -center_z],
                [0, 0, 0, 1],
            ]
        )

        ## Finally, extrinsics matrix. Order matters - do trans then rot
        ext_matrix = rot_matrix @ trans_matrix

        return ext_matrix
