# -*- coding: utf-8 -*-
"""
Analysis and visualization functions for platooning behavior.

This module provides debugging and statistics collection utilities for platoon
operations, including time gap and distance gap tracking between vehicles.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencda.core.plan.planer_debug_helper import PlanDebugHelper
from typing import Optional


class PlatoonDebugHelper(PlanDebugHelper):
    """
    Debug helper for platoon behavior statistics.

    This class collects and stores statistics for platooning operations,
    including time gaps and distance gaps between vehicles over time.

    Parameters
    ----------
    actor_id : int
        The actor ID of the selected vehicle.

    Attributes
    ----------
    time_gap_list : List[List[Optional[float]]]
        Nested list containing intra-time-gap values for all time-steps.
    dist_gap_list : List[List[Optional[float]]]
        Nested list containing distance gap values for all time-steps.
    """

    def __init__(self, actor_id: int):
        super(PlatoonDebugHelper, self).__init__(actor_id)

        self.time_gap_list = [[]]
        self.dist_gap_list = [[]]

    def update(
        self,
        ego_speed: float,
        ttc: float,
        time_gap: Optional[float] = None,
        dist_gap: Optional[float] = None,
    ) -> None:
        """
        Update platoon-related vehicle information.

        Parameters
        ----------
        ego_speed : float
            Ego vehicle speed in m/s.
        ttc : float
            Ego vehicle time-to-collision in seconds.
        time_gap : float, optional
            Ego vehicle time gap with the front vehicle in seconds.
            Default is None.
        dist_gap : float, optional
            Ego vehicle distance gap with front vehicle in meters.
            Default is None.
        """
        super().update(ego_speed, ttc)
        # at the very beginning, the vehicle speed is 0, which causes
        # an infinite time gap.  So we need to filter out the first
        # 100 data points.
        if self.count > 100:
            self.time_gap_list[0].append(time_gap)
            self.dist_gap_list[0].append(dist_gap)
