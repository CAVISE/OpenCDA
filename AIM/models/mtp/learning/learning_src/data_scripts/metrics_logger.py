import os
import matplotlib.pyplot as plt
from collections import deque
import logging


class MetricLogger:
    """
    logger for tracking and visualizing training metrics
    """

    def __init__(self, logfile_path: str, plotfile_path: str, enable_plot: bool, max_points: int = 1000) -> None:
        """
        initialize metric logger

        :param logfile_path: path to log file for saving metrics
        :param plotfile_path: path to plot file for saving metric plots
        :param enable_plot: flag to enable metric plotting
        :param max_points: maximum number of metric points to keep in memory
        """
        self.logfile_path = logfile_path
        self.plotfile_path = plotfile_path
        os.makedirs(os.path.dirname(self.logfile_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.plotfile_path), exist_ok=True)

        self.timestamps = deque(maxlen=max_points)
        self.metrics = deque(maxlen=max_points)

        self.enable_plot = enable_plot
        self.plot_inited = False
        self.fig = None
        self.ax = None
        self.logger = logging.getLogger(__name__)

    def init_plot(self) -> None:
        """
        initialize matplotlib plot figure and axes
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def clear_logs(self) -> None:
        """
        clear all logged timestamps and metrics from memory
        """
        self.timestamps.clear()
        self.metrics.clear()

    def clear_file(self) -> None:
        """
        clear log file by truncating it
        """
        open(self.logfile_path, "w").close()

    def add_metric_points(self, epochs: list[int], timestamps: list[float], metrics: list[float]) -> None:
        """
        add metric points to log file and memory

        :param epochs: list of epoch numbers
        :param timestamps: list of timestamp values
        :param metrics: list of metric values
        """
        if not (len(epochs) == len(timestamps) == len(metrics)):
            self.logger.error("METRIC LOG ERROR: lens of given epochs, timestamps, metrics arent the same <<<<<")
            return

        lines = []
        for epoch, timestamp, metric in zip(epochs, timestamps, metrics):
            lines.append(f"{epoch},{timestamp},{metric}\n")
            self.timestamps.append(timestamp)
            self.metrics.append(metric)

        with open(self.logfile_path, "a") as f:
            f.writelines(lines)

    def plot_metric(self) -> None:
        """
        plot metrics and save to file
        """
        if not self.enable_plot:
            return

        if not self.plot_inited:
            self.init_plot()
            self.plot_inited = True

        self.ax.plot(self.timestamps, self.metrics, color="blue")
        self.ax.set_xlabel("timestamp")
        self.ax.set_ylabel("metric")
        self.ax.grid(True)

        self.fig.savefig(self.plotfile_path)
        self.ax.cla()
