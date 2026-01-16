# -*- coding: utf-8 -*-
"""
Kalman Filter implementation for GPS and IMU sensor fusion.

This module implements a Kalman Filter for improved localization by fusing
GPS position measurements with IMU yaw rate data to provide more accurate
state estimation.

Reference
---------
https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>, Xin Xia<x35xia@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
from typing import Tuple
import numpy as np
import numpy.typing as npt


class KalmanFilter(object):
    """
    Kalman Filter implementation for gps and imu.

    Parameters
    ----------
    dt : float
        The step time for kalman filter calculation.

    Attributes
    ----------
    Q : numpy.array
        predict state covariance.

    R : numpy.array
        Observation x,y position covariance.

    time_step : float
        The step time for kalman filter calculation.

    xEst : numpy.array
        Estimated x values.

    PEst : numpy.array
        The estimated P values.
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

        # Observation x,y position covariance
        self.R = np.diag([0.5, 0.5, 0.2]) ** 2

        self.time_step = dt

        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

    def motion_model(
        self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Predict current position and yaw based on
        previous result (X = F * X_prev + B * u).

        Parameters
        ----------
        x : np.array
            [x_prev, y_prev, yaw_prev, v_prev], shape: (4, 1).

        u : np.array
            [v_current, imu_yaw_rate], shape:(2, 1).

        Returns
        -------
        x : np.array
            Predicted state.
        """
        F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])

        B = np.array([[self.time_step * math.cos(x[2, 0]), 0], [self.time_step * math.sin(x[2, 0]), 0], [0.0, self.time_step], [1.0, 0.0]])

        x = F @ x + B @ u

        return x

    def observation_model(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Map state space to measurement space.

        Implements the measurement equation Z_k = H * X_k where H is the
        observation matrix that projects the state to sensor measurements.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            State vector [x, y, yaw, velocity] with shape (4, 1).

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
        Initial state filling.

        Parameters
        ----------
        x : float
            The x coordinate.

        y : float
            The y coordinate.

        heading : float
            The heading direction.

        velocity : float
            The velocity speed.

        """
        self.xEst[0] = x
        self.xEst[1] = y
        self.xEst[2] = heading
        self.xEst[3] = velocity

    def run_step(
        self, x: float, y: float, heading: float, velocity: float, yaw_rate_imu: float
    ) -> Tuple[float, float, float, float]:
        """
        Apply KF on current measurement and previous prediction.

        Parameters
        ----------
        x : float
            x(esu) coordinate from gnss sensor at current timestamp

        y : float
            y(esu) coordinate from gnss sensor at current timestamp

        heading : float
            heading direction at current timestamp.

        velocity : float
            current speed.

        yaw_rate_imu : float
            yaw rate rad/s from IMU sensor.

        Returns
        -------
        Xest : np.array
            The corrected x, y, heading, and velocity information.
        """
        # gps observation
        z = np.array([x, y, heading]).reshape(3, 1)
        # velocity and imu yaw rate
        u = np.array([velocity, yaw_rate_imu]).reshape(2, 1)

        # state prediction
        xPred = self.motion_model(self.xEst, u)
        # sensor measurement prediction
        zPred = self.observation_model(xPred)
        y = z - zPred

        # projection matrix
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        # prediction matrix
        F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])

        PPred = F @ self.PEst @ F.T + self.Q
        S = np.linalg.inv(H @ PPred @ H.T + self.R)
        K = PPred @ H.T @ S

        self.xEst = xPred + K @ y
        self.PEst = K @ H @ PPred

        return self.xEst[0][0], self.xEst[1][0], self.xEst[2][0], self.xEst[3][0]
