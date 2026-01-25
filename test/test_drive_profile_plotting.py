"""
Unit tests for drive profile plotting functionality.

This module contains test cases for the drive profile plotting functions
in opencda.core.plan.drive_profile_plotting module.
"""

import os
import sys
import unittest


# temporary solution for relative imports in case opencda is not installed
# if opencda is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


from opencda.core.plan.drive_profile_plotting import draw_sub_plot


class TestDriveProfile(unittest.TestCase):
    """
    Test suite for drive profile plotting functions.

    Tests the visualization of vehicle driving profiles including speed,
    acceleration, and other metrics over time.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures.

        Creates mock data lists representing vehicle metrics for testing
        subplot generation.

        Returns
        -------
        None
        """
        self.mock_list = [[23, 25, 25, 44, 66], [44, 55, 25, 22, 33]]

    def test_sub_plot(self) -> None:
        """
        Test subplot generation with multiple metric lists.

        Verifies that the draw_sub_plot function successfully creates
        visualizations when provided with multiple data series.

        Returns
        -------
        None
        """
        assert draw_sub_plot(self.mock_list, self.mock_list, self.mock_list, self.mock_list, self.mock_list)
