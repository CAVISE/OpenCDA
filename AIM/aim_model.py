from abc import ABC, abstractmethod
from typing import List
import torch
from huggingface_hub import PyTorchModelHubMixin
import inspect

from .registry import ModelRegistry


class AIMModelWrapper(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            ModelRegistry.register_model_wrapper(cls)

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, features: List, target_agent_ids: List[str]):
        pass


class AIMModel(ABC, torch.nn.Module, PyTorchModelHubMixin):
    model_wrapper: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            ModelRegistry.register_model(cls)


class MTPModel(AIMModel):
    model_wrapper = "MTP"

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
