"""
Analysis + Visualization functions for planning
"""

from typing import Any, List, Tuple
import warnings

import matplotlib.pyplot as plt

import opencda.core.plan.drive_profile_plotting as open_plt
from opencda.core.plan.metrics.base_metric import BaseMetric
from opencda.core.plan.metrics.dynamics_metric import DynamicsMetric
from opencda.core.plan.metrics.ttc_metric import TtcMetric


class PlanDebugHelper(object):
    """
    Manager class that holds multiple metrics.

    Parameters
    ----------
    actor_id : int
        The actor ID of the target vehicle.
    warmup_steps : int
        The number of steps to ignore at the beginning.
    """

    def __init__(self, actor_id: int, warmup_steps: int = 100):
        self.actor_id = actor_id

        # Initialize metrics
        self.dynamics_metric = DynamicsMetric(warmup_steps)
        self.ttc_metric = TtcMetric(warmup_steps)

        # List of active metrics
        self.metrics: List[BaseMetric] = [self.dynamics_metric, self.ttc_metric]

    def update(self, **kwargs: Any) -> None:
        """
        Update all metrics with new data.

        Parameters
        ----------
        **kwargs : Any
            Dictionary containing simulation state (ego_speed, ttc, etc.)
        """
        for metric in self.metrics:
            metric.update(**kwargs)

    def evaluate(self) -> Tuple[plt.Figure, str]:
        """
        Evaluate the target vehicle and visulize the plot.

        Returns
        -------
        figure : matplotlib.pyplot.figure
            The target vehicle's planning profile (velocity, acceleration, and ttc).
        perform_txt : str
            The target vehicle's planning profile as text.

        """
        warnings.filterwarnings("ignore")
        # draw speed, acc and ttc plotting
        figure = plt.figure()

        # Visualization logic (Specific to current set of metrics)
        # In a fully generic system, this would be dynamic, but for now we map explicitly
        plt.subplot(311)
        open_plt.draw_velocity_profile_single_plot([self.dynamics_metric.speed_list])

        plt.subplot(312)
        open_plt.draw_acceleration_profile_single_plot([self.dynamics_metric.acc_list])

        plt.subplot(313)
        self.ttc_metric.visualize(plt.gca())

        figure.suptitle("planning profile of actor id %d" % self.actor_id)

        perform_txt = ""
        for metric in self.metrics:
            perform_txt += metric.evaluate()

        return figure, perform_txt
