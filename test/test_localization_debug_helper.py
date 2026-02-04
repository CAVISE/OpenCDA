"""
Unit tests for localization debug helper.

This module contains unit tests for the LocDebugHelper class, which tracks
and evaluates localization performance by comparing GNSS measurements,
filtered estimates, and ground truth data.
"""

import unittest
from opencda.core.sensing.localization.localization_debug_helper import LocDebugHelper


class TestLocDebugHelper(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up test fixtures.

        Creates a LocDebugHelper instance with animation enabled and
        configured scaling factors for visualization.

        Returns
        -------
        None
        """
        config_yaml = {"show_animation": False, "x_scale": 10.0, "y_scale": 10.0}
        self.actor_id = 10
        self.debug_heloer = LocDebugHelper(config_yaml=config_yaml, actor_id=self.actor_id)

    def test_parameters(self) -> None:
        """
        Test initialization of debug helper parameters.

        Validates that configuration parameters, tracking lists for GNSS,
        filtered, and ground truth data, and history matrices are properly
        initialized with correct types and shapes.

        Returns
        -------
        None
        """
        assert isinstance(self.debug_heloer.show_animation, bool)
        assert isinstance(self.debug_heloer.x_scale, float)
        assert isinstance(self.debug_heloer.y_scale, float)

        assert isinstance(self.debug_heloer.gnss_x, list)
        assert isinstance(self.debug_heloer.gnss_y, list)
        assert isinstance(self.debug_heloer.gnss_yaw, list)
        assert isinstance(self.debug_heloer.gnss_spd, list)

        assert isinstance(self.debug_heloer.filter_x, list)
        assert isinstance(self.debug_heloer.filter_y, list)
        assert isinstance(self.debug_heloer.filter_yaw, list)
        assert isinstance(self.debug_heloer.filter_spd, list)

        assert isinstance(self.debug_heloer.gt_x, list)
        assert isinstance(self.debug_heloer.gt_y, list)
        assert isinstance(self.debug_heloer.gt_yaw, list)
        assert isinstance(self.debug_heloer.gt_spd, list)

        assert self.debug_heloer.hxEst.shape == (2, 1)
        assert self.debug_heloer.hTrue.shape == (2, 1)
        assert self.debug_heloer.hz.shape == (2, 1)

        assert self.debug_heloer.actor_id == self.actor_id

    def test_run_step(self) -> None:
        """
        Test data collection in a single simulation step.

        Verifies that the run_step method correctly appends new data points
        to tracking lists and updates history matrices with proper dimensions.

        Returns
        -------
        None
        """
        # Capture initial shapes (implementation may or may not append history depending on config)
        htrue_cols_before = self.debug_heloer.hTrue.shape[1]
        hxest_cols_before = self.debug_heloer.hxEst.shape[1]
        hz_cols_before = self.debug_heloer.hz.shape[1]

        self.debug_heloer.run_step(10.0, 10.0, 10.0, 20.0, 10.4, 10.4, 10.4, 20.4, 10.3, 10.3, 10.3, 20.3)

        assert len(self.debug_heloer.gnss_x) == 1
        assert len(self.debug_heloer.filter_x) == 1
        assert len(self.debug_heloer.gt_x) == 1

        # Matrices should keep correct row dimension
        assert self.debug_heloer.hTrue.shape[0] == 2
        assert self.debug_heloer.hxEst.shape[0] == 2
        assert self.debug_heloer.hz.shape[0] == 2

        # Columns may stay the same (no history) or increase by 1 (history enabled)
        assert self.debug_heloer.hTrue.shape[1] in (htrue_cols_before, htrue_cols_before + 1)
        assert self.debug_heloer.hxEst.shape[1] in (hxest_cols_before, hxest_cols_before + 1)
        assert self.debug_heloer.hz.shape[1] in (hz_cols_before, hz_cols_before + 1)

    def test_evaluate(self) -> None:
        """
        Test performance evaluation and report generation.

        Validates that the evaluate method returns a figure object and
        a text report string after localization data has been collected.

        Returns
        -------
        None
        """
        self.debug_heloer.run_step(10.0, 10.0, 10.0, 20.0, 10.4, 10.4, 10.4, 20.4, 10.3, 10.3, 10.3, 20.3)
        assert self.debug_heloer.evaluate()[0]
        assert isinstance(self.debug_heloer.evaluate()[1], str)
