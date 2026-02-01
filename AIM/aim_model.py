from abc import ABC, abstractmethod
from typing import List
from .registry import ModelRegistry


class AIMModel(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelRegistry.register(cls)

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, features: List, target_agent_ids: List[str]):
        pass
