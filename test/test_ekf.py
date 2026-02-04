"""
Unit tests for Extended Kalman Filter localization functionality.

This module contains unit tests for the ExtendedKalmanFilter class used
in vehicle localization systems with nonlinear motion models.
"""

import unittest
from opencda.customize.core.sensing.localization.extented_kalman_filter import ExtentedKalmanFilter


class testKalmanFilter(unittest.TestCase):
    """
    Test suite for ExtendedKalmanFilter class.

    Tests initialization and state estimation for GPS/IMU fusion-based
    vehicle localization using Extended Kalman Filter with nonlinear models.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures.

        Creates an ExtendedKalmanFilter instance with 0.25s time step and
        initializes it with mock position, heading, and velocity data.

        Returns
        -------
        None
        """
        self.dt = 0.25
        self.kf = ExtentedKalmanFilter(self.dt)
        self.kf.run_step_init(10, 10, 90, 20)

    def test_parameters(self) -> None:
        """
        Test Extended Kalman filter parameter initialization.

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

    def test_run_step(self) -> None:
        """
        Test Extended Kalman filter state update.

        Verifies that the run_step method returns four float values
        representing the estimated state (x, y, yaw, velocity) using
        nonlinear prediction and linearized update steps.

        Returns
        -------
        None
        """
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[0], float)
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[1], float)
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[2], float)
        assert isinstance(self.kf.run_step(10, 10, 10, 10, 3)[3], float)


if __name__ == "__main__":
    unittest.main()
