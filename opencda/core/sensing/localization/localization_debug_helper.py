# -*- coding: utf-8 -*-
"""
Visualization tools for localization debugging and analysis.

This module provides utilities for visualizing and evaluating localization
algorithms by comparing GNSS measurements, filtered estimates, and ground truth
data through trajectory plots and error analysis.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib

import matplotlib.pyplot as plt


class LocDebugHelper(object):
    """
    Localization debugging and visualization helper.

    This class helps debug localization algorithms by recording and visualizing
    trajectory data from GNSS measurements, Kalman filter estimates, and ground
    truth. Supports both real-time animation and post-processing analysis.

    Parameters
    ----------
    config_yaml : Dict[str, Any]
        Configuration dictionary containing visualization parameters.
    actor_id : int
        The actor ID for identification in plots.

    Attributes
    ----------
    show_animation : bool
        Whether to display real-time animation.
    x_scale : float
        Scaling factor for x coordinates in visualization.
    y_scale : float
        Scaling factor for y coordinates in visualization.
    gnss_x : List[float]
        Recorded GNSS x coordinates.
    gnss_y : List[float]
        Recorded GNSS y coordinates.
    gnss_yaw : List[float]
        Recorded GNSS yaw angles in degrees.
    gnss_spd : List[float]
        Recorded GNSS speed values in m/s.
    filter_x : List[float]
        Filtered x coordinates.
    filter_y : List[float]
        Filtered y coordinates.
    filter_yaw : List[float]
        Filtered yaw angles in degrees.
    filter_spd : List[float]
        Filtered speed values in m/s.
    gt_x : List[float]
        Ground truth x coordinates.
    gt_y : List[float]
        Ground truth y coordinates.
    gt_yaw : List[float]
        Ground truth yaw angles in degrees.
    gt_spd : List[float]
        Ground truth speed values in m/s.
    hxEst : npt.NDArray[np.float64]
        History of filtered x-y coordinates for animation.
    hTrue : npt.NDArray[np.float64]
        History of ground truth x-y coordinates for animation.
    hz : npt.NDArray[np.float64]
        History of GNSS x-y coordinates for animation.
    actor_id : int
        Actor identifier for labeling plots.
    """

    def __init__(self, config_yaml: Dict[str, Any], actor_id: int):
        self.show_animation = config_yaml["show_animation"]
        self.x_scale = config_yaml["x_scale"]
        self.y_scale = config_yaml["y_scale"]

        # off-line plotting
        self.gnss_x = []
        self.gnss_y = []
        self.gnss_yaw = []
        self.gnss_spd = []

        self.filter_x = []
        self.filter_y = []
        self.filter_yaw = []
        self.filter_spd = []

        self.gt_x = []
        self.gt_y = []
        self.gt_yaw = []
        self.gt_spd = []

        # online animation
        # filtered x y coordinates
        self.hxEst = np.zeros((2, 1))
        # gt x y coordinates
        self.hTrue = np.zeros((2, 1))
        # gnss x y coordinates
        self.hz = np.zeros((2, 1))

        self.actor_id = actor_id

    def run_step(
        self,
        gnss_x: float,
        gnss_y: float,
        gnss_yaw: float,
        gnss_spd: float,
        filter_x: float,
        filter_y: float,
        filter_yaw: float,
        filter_spd: float,
        gt_x: float,
        gt_y: float,
        gt_yaw: float,
        gt_spd: float,
    ) -> None:
        """
        Record and optionally animate one step of localization data.

        Parameters
        ----------
        gnss_x : float
            GNSS detected x coordinate.
        gnss_y : float
            GNSS detected y coordinate.
        gnss_yaw : float
            GNSS detected yaw angle in degrees.
        gnss_spd : float
            GNSS detected speed in km/h.
        filter_x : float
            Filtered x coordinate.
        filter_y : float
            Filtered y coordinate.
        filter_yaw : float
            Filtered yaw angle in degrees.
        filter_spd : float
            Filtered speed in km/h.
        gt_x : float
            Ground truth x coordinate.
        gt_y : float
            Ground truth y coordinate.
        gt_yaw : float
            Ground truth yaw angle in degrees.
        gt_spd : float
            Ground truth speed in km/h.
        """
        self.gnss_x.append(gnss_x)
        self.gnss_y.append(gnss_y)
        self.gnss_yaw.append(gnss_yaw)
        self.gnss_spd.append(gnss_spd / 3.6)

        self.filter_x.append(filter_x)
        self.filter_y.append(filter_y)
        self.filter_yaw.append(filter_yaw)
        self.filter_spd.append(filter_spd / 3.6)

        self.gt_x.append(gt_x)
        self.gt_y.append(gt_y)
        self.gt_yaw.append(gt_yaw)
        self.gt_spd.append(gt_spd / 3.6)

        if self.show_animation:
            # call backend setting here to solve the conflict between cv2 pyqt5
            # and pyplot qtagg
            try:
                matplotlib.use("TkAgg")
            except ImportError:
                pass
            xEst = np.array([filter_x, filter_y]).reshape(2, 1)
            zTrue = np.array([gt_x, gt_y]).reshape(2, 1)
            z = np.array([gnss_x, gnss_y]).reshape(2, 1)

            self.hxEst = np.hstack((self.hxEst, xEst))
            self.hz = np.hstack((self.hz, z))
            self.hTrue = np.hstack((self.hTrue, zTrue))

            plt.cla()
            plt.title("actor id %d localization trajectory" % self.actor_id)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect("key_release_event", lambda event: [plt.close() if event.key == "escape" else None])

            plt.plot(self.hTrue[0, 1:].flatten() * self.x_scale, self.hTrue[1, 1:].flatten() * self.y_scale, "-b", label="groundtruth")
            plt.plot(self.hz[0, 1:] * self.x_scale, self.hz[1, 1:] * self.y_scale, ".g", label="gnss noise data")
            plt.plot(self.hxEst[0, 1:].flatten() * self.x_scale, self.hxEst[1, 1:].flatten() * self.y_scale, "-r", label="kf result")

            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

    def evaluate(self) -> Tuple[plt.Figure, str]:
        """
        Generate visualization and compute localization error statistics.

        Creates a comprehensive plot with trajectory, yaw, speed, and error curves.
        Computes mean absolute errors for GNSS and filtered estimates.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure containing six subplots showing localization performance.
        perform_txt : str
            Text summary of mean errors for GNSS and filtered data.
        """
        figure, axis = plt.subplots(3, 2)
        figure.set_size_inches(16, 12)

        # Plot positions
        axis[0, 0].plot(self.gnss_x, self.gnss_y, ".g", label="gnss")
        axis[0, 0].plot(self.gt_x, self.gt_y, ".b", label="gt")
        axis[0, 0].plot(self.filter_x, self.filter_y, ".r", label="filter")
        axis[0, 0].legend()
        axis[0, 0].set_title("x-y coordinates plotting")

        # Plot yaw
        axis[0, 1].plot(np.arange(len(self.gnss_yaw)), self.gnss_yaw, ".g", label="gnss")
        axis[0, 1].plot(np.arange(len(self.gt_yaw)), self.gt_yaw, ".b", label="gt")
        axis[0, 1].plot(np.arange(len(self.filter_yaw)), self.filter_yaw, ".r", label="filter")
        axis[0, 1].legend()
        axis[0, 1].set_title("yaw angle (degree) plotting")

        # Plot speed
        axis[1, 0].plot(np.arange(len(self.gnss_spd)), self.gnss_spd, ".g", label="gnss")
        axis[1, 0].plot(np.arange(len(self.gt_spd)), self.gt_spd, ".b", label="gt")
        axis[1, 0].plot(np.arange(len(self.filter_spd)), self.filter_spd, ".r", label="filter")
        axis[1, 0].legend()
        axis[1, 0].set_title("speed (m/s) plotting")

        # Plot x error
        axis[1, 1].plot(np.arange(len(self.gnss_x)), np.array(self.gt_x) - np.array(self.gnss_x), "-g", label="gnss")
        axis[1, 1].plot(np.arange(len(self.filter_x)), np.array(self.gt_x) - np.array(self.filter_x), "-r", label="filter")
        axis[1, 1].legend()
        axis[1, 1].set_title("error curve on x coordinates")

        # Plot y error
        axis[2, 0].plot(np.arange(len(self.gnss_y)), np.array(self.gt_y) - np.array(self.gnss_y), "-g", label="gnss")
        axis[2, 0].plot(np.arange(len(self.filter_y)), np.array(self.gt_y) - np.array(self.filter_y), "-r", label="filter")
        axis[2, 0].legend()
        axis[2, 0].set_title("error curve on y coordinates")

        # Plot yaw error
        axis[2, 1].plot(np.arange(len(self.gnss_yaw)), np.array(self.gt_yaw) - np.array(self.gnss_yaw), "-g", label="gnss")
        axis[2, 1].plot(np.arange(len(self.filter_yaw)), np.array(self.gt_yaw) - np.array(self.filter_yaw), "-r", label="filter")
        axis[2, 1].legend()
        axis[2, 1].set_title("error curve on yaw angle")

        figure.suptitle(f"Localization plotting of actor id {self.actor_id}")

        perform_txt = ""
        perform_txt += self._format_mean_error("GNSS raw data", self.gt_x, self.gnss_x, self.gt_y, self.gnss_y, self.gt_yaw, self.gnss_yaw)
        perform_txt += self._format_mean_error("data fusion", self.gt_x, self.filter_x, self.gt_y, self.filter_y, self.gt_yaw, self.filter_yaw)

        return figure, perform_txt

    def _safe_mean_error(self, a: List[float], b: List[float]) -> float:
        """
        Compute mean absolute error safely handling empty lists.

        Parameters
        ----------
        a : List[float]
            First list of values.
        b : List[float]
            Second list of values.

        Returns
        -------
        float
            Mean absolute error, or NaN if either list is empty.
        """
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        return np.mean(np.abs(np.array(a) - np.array(b)))

    def _format_mean_error(
        self,
        label: str,
        x1: List[float],
        x2: List[float],
        y1: List[float],
        y2: List[float],
        yaw1: List[float],
        yaw2: List[float],
    ) -> str:
        """
        Format mean error statistics as a text string.

        Parameters
        ----------
        label : str
            Label describing the data source (e.g., "GNSS raw data").
        x1 : List[float]
            Ground truth x coordinates.
        x2 : List[float]
            Estimated x coordinates.
        y1 : List[float]
            Ground truth y coordinates.
        y2 : List[float]
            Estimated y coordinates.
        yaw1 : List[float]
            Ground truth yaw angles.
        yaw2 : List[float]
            Estimated yaw angles.

        Returns
        -------
        str
            Formatted string with mean errors for x, y, and yaw.
        """
        x_error = self._safe_mean_error(x1, x2)
        y_error = self._safe_mean_error(y1, y2)
        yaw_error = self._safe_mean_error(yaw1, yaw2)
        return f"mean error for {label} on x-axis: {x_error:.3f} (m), on y-axis: {y_error:.3f} (m), on yaw: {yaw_error:.3f} (°)\n"
