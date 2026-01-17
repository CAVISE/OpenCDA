# -*- coding: utf-8 -*-
"""
Extended Kalman Filter for GPS and IMU sensor fusion.

This module implements an Extended Kalman Filter (EKF) for improved vehicle
localization by fusing GPS position measurements with IMU angular rate data.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
from typing import Tuple
import numpy as np
import numpy.typing as npt


class ExtentedKalmanFilter(object):
    """
    Extended Kalman Filter for GPS and IMU sensor fusion.

    Implements EKF for state estimation of vehicle position, heading, and
    velocity using GPS measurements and IMU yaw rate data.

    Parameters
    ----------
    dt : float
        Time step for Kalman filter calculations in seconds.

    Attributes
    ----------
    Q : npt.NDArray[np.float64]
        Process noise covariance matrix (4x4) for state prediction.
    R : npt.NDArray[np.float64]
        Measurement noise covariance matrix (3x3) for observations.
    time_step : float
        Time step for filter calculations in seconds.
    xEst : npt.NDArray[np.float64]
        Estimated state vector [x, y, yaw, velocity] with shape (4, 1).
    PEst : npt.NDArray[np.float64]
        Estimated state covariance matrix (4x4).
    """

    def __init__(self, dt: float):
        self.Q = (
            np.diag(
                [
                    0.2,  # variance of location on x-axis
                    0.2,  # variance of location on y-axis
                    np.deg2rad(0.1),  # variance of yaw angle
                    0.001,  # variance of velocity
                ]
            )
            ** 2
        )  # predict state covariance

        self.R = np.diag([0.5, 0.5, 0.2]) ** 2

        self.time_step = dt

        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

    def motion_model(self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predict current state based on previous state and control input.

        Uses the motion model: X = F * X_prev + B * u

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Previous state vector [x_prev, y_prev, yaw_prev, v_prev] with shape (4, 1).
        u : npt.NDArray[np.float64]
            Control input vector [v_current, imu_yaw_rate] with shape (2, 1).

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted state vector with shape (4, 1).
        """
        F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])

        B = np.array([[self.time_step * math.cos(x[2, 0]), 0], [self.time_step * math.sin(x[2, 0]), 0], [0.0, self.time_step], [1.0, 0.0]])

        x = F @ x + B @ u

        return x

    def jacob_f(self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate Jacobian matrix of the motion model.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Current state vector [x, y, yaw, v] with shape (4, 1).
        u : npt.NDArray[np.float64]
            Control input vector [v_current, imu_yaw_rate] with shape (2, 1).

        Returns
        -------
        npt.NDArray[np.float64]
            Jacobian matrix of motion model with shape (4, 4).
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array(
            [
                [1.0, 0.0, -self.time_step * v * math.sin(yaw), self.time_step * math.cos(yaw)],
                [0.0, 1.0, self.time_step * v * math.cos(yaw), self.time_step * math.sin(yaw)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return jF

    def observation_model(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Project state vector to measurement space.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            State vector [x, y, yaw, v] with shape (4, 1).

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted measurement vector [x, y, yaw] with shape (3, 1).
        """
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        z = H @ x

        return z

    def run_step_init(self, x: float, y: float, heading: float, velocity: float) -> None:
        """
        Initialize filter state with initial measurements.

        Parameters
        ----------
        x : float
            Initial x coordinate in meters.
        y : float
            Initial y coordinate in meters.
        heading : float
            Initial heading direction in radians.
        velocity : float
            Initial velocity in m/s.
        """
        self.xEst[0] = x
        self.xEst[1] = y
        self.xEst[2] = heading
        self.xEst[3] = velocity

    def run_step(self, x: float, y: float, heading: float, velocity: float, yaw_rate_imu: float) -> Tuple[float, float, float, float]:
        """
        Execute one EKF prediction and correction step.

        Performs Extended Kalman Filter prediction using motion model and
        correction using GPS and IMU measurements.

        Parameters
        ----------
        x : float
            X coordinate from GNSS sensor in meters.
        y : float
            Y coordinate from GNSS sensor in meters.
        heading : float
            Heading direction in radians.
        velocity : float
            Current speed in m/s.
        yaw_rate_imu : float
            Yaw rate from IMU sensor in rad/s.

        Returns
        -------
        x_est : float
            Corrected x coordinate in meters.
        y_est : float
            Corrected y coordinate in meters.
        heading_est : float
            Corrected heading in radians.
        velocity_est : float
            Corrected velocity in m/s.
        """

        # gps observation
        z = np.array([x, y, heading]).reshape(3, 1)
        # velocity and imu yaw rate
        u = np.array([velocity, yaw_rate_imu]).reshape(2, 1)

        # EKF starts
        xPred = self.motion_model(self.xEst, u)
        jF = self.jacob_f(self.xEst, u)
        PPred = jF @ self.PEst @ jF.T + self.Q

        # Jacobian of Observation Model
        jH = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        self.xEst = xPred + K @ y
        self.PEst = (np.eye(len(self.xEst)) - K @ jH) @ PPred

        return self.xEst[0][0], self.xEst[1][0], self.xEst[2][0], self.xEst[3][0]
