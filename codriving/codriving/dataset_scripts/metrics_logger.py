import os
import matplotlib.pyplot as plt
from collections import deque
import logging


class MetricLogger:
  def __init__(self, logfile_path: str, plotfile_path: str, enable_plot: bool, max_points=1000):
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

  def init_plot(self):
    self.fig, self.ax = plt.subplots(figsize=(8, 6))

  def clear_logs(self):
    self.timestamps.clear()
    self.metrics.clear()

  def clear_file(self):
    open(self.logfile_path, 'w').close()

  def add_metric_points(self, epochs: list[int], timestamps: list[float], metrics: list[float]):
    if not (len(epochs) == len(timestamps) == len(metrics)):
      self.logger.error('METRIC LOG ERROR: lens of given epochs, timestamps, metrics arent the same <<<<<')
      return

    lines = []
    for epoch, timestamp, metric in zip(epochs, timestamps, metrics):
      lines.append(f"{epoch},{timestamp},{metric}\n")
      self.timestamps.append(timestamp)
      self.metrics.append(metric)

    with open(self.logfile_path, 'a') as f:
      f.writelines(lines)

  def plot_metric(self):
    if not self.enable_plot:
      return

    if not self.plot_inited:
      self.init_plot()
      self.plot_inited = True

    self.ax.plot(self.timestamps, self.metrics, color='blue')
    self.ax.set_xlabel('timestamp')
    self.ax.set_ylabel('metric')
    self.ax.grid(True)

    self.fig.savefig(self.plotfile_path)
    self.ax.cla()
