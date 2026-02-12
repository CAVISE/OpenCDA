from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
import numpy.typing as npt
import torch
from .registry import ModelRegistry


class AIMModel(ABC):
    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        ModelRegistry.register(cls)

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def predict(self, features: npt.NDArray[np.float64], target_agent_ids: List[str]) -> torch.Tensor:
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
