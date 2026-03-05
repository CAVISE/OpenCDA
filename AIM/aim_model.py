from abc import ABC, abstractmethod
from typing import Any, List
from .registry import ModelRegistry


class AIMModel(ABC):
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
        ModelRegistry.register(cls)

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
    def predict(self, features: List[Any], target_agent_ids: List[str]) -> Any:
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
