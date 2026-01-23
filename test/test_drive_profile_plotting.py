"""
Unit tests for drive_profile_plotting module.
Smoke tests to verify plotting functions run without errors.
"""

import os
import sys
import unittest

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for CI/headless testing
import matplotlib.pyplot as plt

# temporary solution for relative imports in case opencda is not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from opencda.core.plan.drive_profile_plotting import (
    draw_velocity_profile_single_plot,
    draw_acceleration_profile_single_plot,
    draw_ttc_profile_single_plot,
    draw_time_gap_profile_singel_plot,
    draw_dist_gap_profile_singel_plot,
)


class TestDriveProfilePlotting(unittest.TestCase):
    """Smoke tests for drive profile plotting functions."""

    def setUp(self):
        """Set up test data."""
        self.mock_velocity = [[23, 25, 25, 30, 28], [20, 22, 24, 26, 28]]
        self.mock_acceleration = [[0.5, 1.0, -0.5, 0.0, 0.2], [0.3, -0.2, 0.1, 0.0, -0.1]]
        self.mock_ttc = [[5, 8, 10, 15, 20], [6, 9, 12, 18, 25]]
        self.mock_time_gap = [[0.8, 0.9, 1.0, 1.1, 1.2], [0.7, 0.8, 0.9, 1.0, 1.1]]
        self.mock_dist_gap = [[15, 18, 20, 22, 25], [12, 15, 18, 20, 22]]

    def tearDown(self):
        """Clean up matplotlib figures after each test."""
        plt.close('all')

    def _assert_figure_created(self):
        """Helper: verify at least one axis exists on current figure."""
        fig = plt.gcf()
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.axes), 1, "Expected at least one axis on figure")

    def test_draw_velocity_profile_single_plot(self):
        """Test velocity profile plotting runs without error."""
        draw_velocity_profile_single_plot(self.mock_velocity)
        self._assert_figure_created()

    def test_draw_acceleration_profile_single_plot(self):
        """Test acceleration profile plotting runs without error."""
        draw_acceleration_profile_single_plot(self.mock_acceleration)
        self._assert_figure_created()

    def test_draw_ttc_profile_single_plot(self):
        """Test TTC profile plotting runs without error."""
        draw_ttc_profile_single_plot(self.mock_ttc)
        self._assert_figure_created()

    def test_draw_time_gap_profile_single_plot(self):
        """Test time gap profile plotting runs without error."""
        # Note: function has typo "singel" in production code
        draw_time_gap_profile_singel_plot(self.mock_time_gap)
        self._assert_figure_created()

    def test_draw_dist_gap_profile_single_plot(self):
        """Test distance gap profile plotting runs without error."""
        # Note: function has typo "singel" in production code
        draw_dist_gap_profile_singel_plot(self.mock_dist_gap)
        self._assert_figure_created()

    def test_single_profile(self):
        """Test plotting with single profile list."""
        single_velocity = [[10, 15, 20, 25, 30]]
        draw_velocity_profile_single_plot(single_velocity)
        self._assert_figure_created()


if __name__ == "__main__":
    unittest.main()