from abc import ABC, abstractmethod
from typing import Any
import matplotlib.pyplot as plt


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning of the simulation.
    """

    def __init__(self, warmup_steps: int = 100):
        self.warmup_steps = warmup_steps
        self.count = 0
        self.data = []

    def update(self, **kwargs: Any) -> None:
        """
        Update metric state. Should be implemented by subclasses.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments containing simulation data.
        """
        self.count += 1
        if self.count > self.warmup_steps:
            self._process_data(**kwargs)

    @abstractmethod
    def _process_data(self, **kwargs: Any) -> None:
        """Process the data after warmup period."""
        pass

    @abstractmethod
    def evaluate(self) -> str:
        """Return statistics string."""
        pass

    @abstractmethod
    def visualize(self, ax: plt.Axes) -> None:
        """Draw plot on the given matplotlib axis."""
        pass
