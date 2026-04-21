from abc import ABC, abstractmethod
from typing import Any, List
import torch
from huggingface_hub import PyTorchModelHubMixin
import inspect

from .registry import ModelRegistry


class AIMModelWrapper(ABC):
    """
    Abstract base class for all AIM models.

    All subclasses are automatically registered in ModelRegistry.
    Every model must implement the `predict` method.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically register subclasses in ModelRegistry.
        """
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            ModelRegistry.register_model_wrapper(cls)

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        **kwargs : Any
            Model-specific configuration parameters.
        """
        pass

    @abstractmethod
    def predict(
        self,
        features: Any,
        target_agent_ids: List[str],
    ) -> Any:
        """
        Perform model inference.

        Parameters
        ----------
        features : list
            Input feature container.
        target_agent_ids : list[str]
            Target agent identifiers (reserved for future use).

        Returns
        -------
        Any
            Model-specific predictions.
        """
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
