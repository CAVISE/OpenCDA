"""
Visualization tools for localization
"""

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import matplotlib

import matplotlib.pyplot as plt

FloatArray: TypeAlias = npt.NDArray[np.float64]


class LocDebugHelper(object):
    """
    This class aims to help users debugging their localization algorithms.
    Users can apply this class to draw the x, y coordinate
    trajectory, yaw angle and vehicle speed from GNSS raw measurements,
    Kalman filter, and the groundtruth measurements.
    Error plotting is also enabled.

    Attributes
        show_animation : boolean
            Indicator of whether to visulize animtion.
        x_scale : float
            The scale of x coordinates.
        y_scale : float
            The scale of y coordinates.
        gnss_x : list
            The list of recorded gnss x coordinates.
        gnss_y : list
            The list of recorded gnss y coordinates.
        gnss_yaw : list
            The list of recorded gnss yaw angles.
        gnss_speed : list
            The list of recorded gnss speed values.
        filter_x : list
            The list of filtered x coordinates.
        filter_y : list
            The list of filtered y coordinates.
        filter_yaw : list
            The list of filtered yaw angles.
        filter_speed : list
            The list of filtered speed values.
        gt_x : list
            The list of ground truth x coordinates.
        gt_y : list
            The list of ground truth y coordinates.
        gt_yaw : list
            The list of ground truth yaw angles.
        gt_speed : list
            The list of ground truth speed values.
        hxEst : list
            The filtered x y coordinates.
        hTrue : list
            The true x y coordinates.
        hz : list
            The gnss detected x y coordinates.
        actor_id : int
            The list of ground truth speed values.
    """

    def __init__(self, config_yaml: dict[str, Any], actor_id: int) -> None:
        self.show_animation: bool = config_yaml["show_animation"]
        self.x_scale: float = config_yaml["x_scale"]
        self.y_scale: float = config_yaml["y_scale"]

        # off-line plotting
        self.gnss_x: list[float] = []
        self.gnss_y: list[float] = []
        self.gnss_yaw: list[float] = []
        self.gnss_spd: list[float] = []

        self.filter_x: list[float] = []
        self.filter_y: list[float] = []
        self.filter_yaw: list[float] = []
        self.filter_spd: list[float] = []

        self.gt_x: list[float] = []
        self.gt_y: list[float] = []
        self.gt_yaw: list[float] = []
        self.gt_spd: list[float] = []

        # online animation
        # filtered x y coordinates
        self.hxEst: FloatArray = np.zeros((2, 1))
        # gt x y coordinates
        self.hTrue: FloatArray = np.zeros((2, 1))
        # gnss x y coordinates
        self.hz: FloatArray = np.zeros((2, 1))

        self.actor_id: int = actor_id

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
        Run a single step for DebugHelper to save and animate(optional)
        the localization data.

        Args:
            -gnss_x (float): GNSS detected x coordinate.
            -gnss_y (float): GNSS detected y coordinate.
            -gnss_yaw (float): GNSS detected yaw angle.
            -gnss_spd (float): GNSS detected speed value.
            -filter_x (float): Filtered x coordinates.
            -filter_y (float): Filtered y coordinates.
            -filter_yaw (float): Filtered yaw angle.
            -filter_spd (float): Filtered speed value.
            -gt_x (float): The ground truth x coordinate.
            -gt_y (float): The ground truth y coordinate.
            -gt_yaw (float): The ground truth yaw angle.
            -gt_spd (float): The ground truth speed value.

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
            def close_on_escape(event: Any) -> None:
                if getattr(event, "key", None) == "escape":
                    plt.close()

            plt.gcf().canvas.mpl_connect("key_release_event", close_on_escape)

            plt.plot(self.hTrue[0, 1:].flatten() * self.x_scale, self.hTrue[1, 1:].flatten() * self.y_scale, "-b", label="groundtruth")
            plt.plot(self.hz[0, 1:] * self.x_scale, self.hz[1, 1:] * self.y_scale, ".g", label="gnss noise data")
            plt.plot(self.hxEst[0, 1:].flatten() * self.x_scale, self.hxEst[1, 1:].flatten() * self.y_scale, "-r", label="kf result")

            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)
