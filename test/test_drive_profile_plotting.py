"""
Unit test for
"""

import unittest
from opencda.core.plan.drive_profile_plotting import draw_sub_plot


class TestDriveProfile(unittest.TestCase):
    def setUp(self):
        self.mock_list = [[23, 25, 25, 44, 66], [44, 55, 25, 22, 33]]

    def test_sub_plot(self):
        assert draw_sub_plot(self.mock_list, self.mock_list, self.mock_list, self.mock_list, self.mock_list)
