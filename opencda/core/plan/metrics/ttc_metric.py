from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt

import opencda.core.plan.drive_profile_plotting as open_plt
from opencda.core.plan.metrics.base_metric import BaseMetric


class TtcMetric(BaseMetric):
    """
    Metric for Time To Collision (TTC).

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning.
    """

    def __init__(self, warmup_steps: int = 100):
        super().__init__(warmup_steps)
        self.ttc_list: List[float] = []

    def _process_data(self, **kwargs: Any) -> None:
        ttc = kwargs.get("ttc", 1000.0)
        self.ttc_list.append(ttc)

    def evaluate(self) -> str:
        if not self.ttc_list:
            return "No TTC data\n"

        ttc_array = np.array(self.ttc_list)
        ttc_array = ttc_array[ttc_array < 1000]
        if len(ttc_array) == 0:
            return "TTC average: N/A, TTC std: N/A \n"

        ttc_avg = np.mean(ttc_array)
        ttc_std = np.std(ttc_array)
        return f"TTC average: {ttc_avg:f} (m/s), TTC std: {ttc_std:f} (m/s) \n"

    def visualize(self, ax: plt.Axes) -> None:
        open_plt.draw_ttc_profile_single_plot([self.ttc_list])
