# -*- coding: utf-8 -*-
"""
Unit test for Localization DebugHelper.

This module contains unit tests for the PlatoonDebugHelper class, which
tracks and evaluates platooning performance metrics such as speed, acceleration,
time-to-collision, and inter-vehicle gaps.
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

from opencda.core.application.platooning.platoon_debug_helper import PlatoonDebugHelper


class TestPlanDebugHelper(unittest.TestCase):
    """
    Test suite for PlatoonDebugHelper class.

    Tests initialization, metric tracking updates, and performance evaluation
    for platooning applications.
    """

    def setUp(self):
        """
        Set up test fixtures.

        Creates a PlatoonDebugHelper instance with mock actor ID and
        initializes the frame counter.

        Returns
        -------
        None
        """
        self.actor_id = 10
        self.platoon_debug_helper = PlatoonDebugHelper(actor_id=self.actor_id)
        self.platoon_debug_helper.count = 100

    def test_parameters(self):
        """
        Test initialization of metric tracking lists.

        Validates that all metric lists (speed, acceleration, TTC, time gap,
        distance gap) are properly initialized as list types.

        Returns
        -------
        None
        """
        assert isinstance(self.platoon_debug_helper.speed_list[0], list)
        assert isinstance(self.platoon_debug_helper.acc_list[0], list)
        assert isinstance(self.platoon_debug_helper.ttc_list[0], list)
        assert isinstance(self.platoon_debug_helper.time_gap_list[0], list)
        assert isinstance(self.platoon_debug_helper.dist_gap_list[0], list)

    def test_update(self):
        """
        Test metric update functionality.

        Verifies that the update method correctly increments frame count
        and maintains proper list lengths for all tracked metrics.

        Returns
        -------
        None
        """
        self.platoon_debug_helper.update(90, 2, 0.8, 10)

        assert self.platoon_debug_helper.count == 101
        assert len(self.platoon_debug_helper.speed_list) == 1
        assert len(self.platoon_debug_helper.acc_list) == 1
        assert len(self.platoon_debug_helper.ttc_list) == 1
        assert len(self.platoon_debug_helper.time_gap_list) == 1
        assert len(self.platoon_debug_helper.dist_gap_list) == 1

    def test_evaluate(self):
        """
        Test performance evaluation and report generation.

        Validates that the evaluate method returns both a figure object
        and a text report string after metrics have been collected.

        Returns
        -------
        None
        """
        self.platoon_debug_helper.update(90, 2, 0.8, 10)
        figure, txt = self.platoon_debug_helper.evaluate()
        assert figure and isinstance(txt, str)
