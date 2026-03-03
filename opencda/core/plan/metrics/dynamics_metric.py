from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt

import opencda.core.plan.drive_profile_plotting as open_plt
from opencda.core.plan.metrics.base_metric import BaseMetric


class DynamicsMetric(BaseMetric):
    """
    Metric for Speed and Acceleration.

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning.
    dt : float
        Time step interval for acceleration calculation.
    """

    def __init__(self, warmup_steps: int = 100, dt: float = 0.05):
        super().__init__(warmup_steps)
        self.dt = dt
        self.speed_list: List[float] = []
        self.acc_list: List[float] = []

    def _process_data(self, **kwargs: Any) -> None:
        ego_speed = kwargs.get("ego_speed", 0.0)

        self.speed_list.append(ego_speed / 3.6)
        if len(self.speed_list) <= 1:
            self.acc_list.append(0.0)
        else:
            self.acc_list.append((self.speed_list[-1] - self.speed_list[-2]) / self.dt)

    def evaluate(self) -> str:
        if not self.speed_list:
            return "No dynamics data\n"

        spd_avg = np.mean(np.array(self.speed_list))
        spd_std = np.std(np.array(self.speed_list))
        acc_avg = np.mean(np.array(self.acc_list))
        acc_std = np.std(np.array(self.acc_list))

        return (
            f"Speed average: {spd_avg:f} (m/s), Speed std: {spd_std:f} (m/s) \n"
            f"Acceleration average: {acc_avg:f} (m/s), Acceleration std: {acc_std:f} (m/s) \n"
        )

    def visualize(self, ax: plt.Axes) -> None:
        # Dynamics metric needs 2 subplots
        open_plt.draw_velocity_profile_single_plot([self.speed_list])
        pass
