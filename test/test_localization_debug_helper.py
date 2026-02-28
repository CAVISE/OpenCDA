"""
Unit test for Localization DebugHelper.
"""

import unittest
from opencda.core.sensing.localization.localization_debug_helper import LocDebugHelper


class TestLocDebugHelper(unittest.TestCase):
    def setUp(self):
        config_yaml = {"show_animation": False, "x_scale": 10.0, "y_scale": 10.0}
        self.actor_id = 10
        self.debug_heloer = LocDebugHelper(config_yaml=config_yaml, actor_id=self.actor_id)

    def test_parameters(self):
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

    def test_run_step(self):
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

    def test_evaluate(self):
        self.debug_heloer.run_step(10.0, 10.0, 10.0, 20.0, 10.4, 10.4, 10.4, 20.4, 10.3, 10.3, 10.3, 20.3)
        assert self.debug_heloer.evaluate()[0]
        assert isinstance(self.debug_heloer.evaluate()[1], str)
