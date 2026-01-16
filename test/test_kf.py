# -*- coding: utf-8 -*-
"""
Unit tests for Kalman filter localization functionality.

This module contains unit tests for the KalmanFilter class and coordinate
transformation utilities used in vehicle localization systems.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import os
import sys
import unittest


# temporary solution for relative imports in case opencda is not installed
# if opencda is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from opencda.core.sensing.localization.kalman_filter import KalmanFilter
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform


class testKalmanFilter(unittest.TestCase):
    """
    Test suite for KalmanFilter class.

    Tests initialization, state estimation, and coordinate transformation
    for GPS/IMU fusion-based vehicle localization.
    """

    def setUp(self):
        """
        Set up test fixtures.

        Creates a KalmanFilter instance with 0.25s time step and initializes
        it with mock position, heading, and velocity data.

        Returns
        -------
        None
        """
        self.dt = 0.25
        self.kf = KalmanFilter(self.dt)
        self.kf.run_step_init(10, 10, 90, 20)

    def test_parameters(self):
        """
        Test Kalman filter parameter initialization.

        Validates that process noise covariance (Q), measurement noise
        covariance (R), time step, state estimate (xEst), and error
        covariance (PEst) are properly initialized with correct shapes.

        Returns
        -------
        None
        """
        assert hasattr(self.kf, "Q") and self.kf.Q.shape == (4, 4)
        assert hasattr(self.kf, "R") and self.kf.R.shape == (3, 3)
        assert hasattr(self.kf, "time_step") and self.kf.time_step == self.dt
        assert hasattr(self.kf, "xEst") and self.kf.xEst.shape == (4, 1)
        assert hasattr(self.kf, "PEst") and self.kf.PEst.shape == (4, 4)

    def test_run_step(self):
        """
        Test Kalman filter state update.

        Verifies that the run_step method returns four float values
        representing the estimated state (x, y, yaw, velocity).

        Returns
        -------
        None
        """
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[0], float)
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[1], float)
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[2], float)
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[3], float)

    def test_geo_to_transform(self):
        """
        Test geographic to local coordinate transformation.

        Validates that GPS coordinates (latitude, longitude) are correctly
        converted to local Cartesian coordinates (x, y, z).

        Returns
        -------
        None
        """
        assert isinstance(geo_to_transform(100, 70, 10, 10, 10, 10)[0], float)
        assert isinstance(geo_to_transform(100, 70, 10, 10, 10, 10)[1], float)
        assert isinstance(geo_to_transform(100, 70, 10.0, 10, 10, 10.0)[2], float)


if __name__ == "__main__":
    unittest.main()
