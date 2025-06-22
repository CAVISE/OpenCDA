# -*- coding: utf-8 -*-
"""
Unit test for
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


from opencda.core.plan.drive_profile_plotting import draw_sub_plot


class TestDriveProfile(unittest.TestCase):
    def setUp(self):
        self.mock_list = [[23, 25, 25, 44, 66], [44, 55, 25, 22, 33]]

    def test_sub_plot(self):
        assert draw_sub_plot(self.mock_list, self.mock_list, self.mock_list, self.mock_list, self.mock_list)
