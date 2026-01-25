from abc import ABC, abstractmethod
from typing import Any, List
import numpy.typing as npt
from .registry import ModelRegistry


class AIMModel(ABC):
    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        ModelRegistry.register(cls)

    @abstractmethod
    def predict(self, features: List[npt.NDArray[float]], target_agent_ids: List[str]) -> npt.NDArray[float]:
        """
        Predict trajectories (or other outputs) for the given agents.

        Parameters
        ----------
        features : NDArray
            Input features for all agents
        target_agent_ids : List[str]
            Identifiers of agents to produce predictions for.
        """
        pass
