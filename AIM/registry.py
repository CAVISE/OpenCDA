import inspect
from typing import Dict, Type, Any, KeysView


class ModelRegistry:
    """
    Global registry for AIM models.

    Stores all non-abstract subclasses of AIMModel and provides
    factory methods for retrieving model instances.
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, model_cls: Type) -> None:
        """
        Register a model class in the global registry.

        Abstract classes are ignored automatically.

        Parameters
        ----------
        model_cls : Type
            Model class to register.

        Raises
        ------
        ValueError
            If a model with the same name already exists.
        """
        if inspect.isabstract(model_cls):
            return

        name: str = model_cls.__name__

        if name in cls._registry:
            raise ValueError(f"Duplicate model name: {name}")

        cls._registry[name] = model_cls

    @classmethod
    def get_model(cls, name: str, **kwargs: Any) -> Any:
        """
        Instantiate a registered model by name.

        Parameters
        ----------
        name : str
            Name of the registered model.
        **kwargs : Any
            Arguments forwarded to the model constructor.

        Returns
        -------
        Any
            Instantiated model.

        Raises
        ------
        KeyError
            If the model name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Unknown model '{name}'. Available: {list(cls._registry)}"
            )

        model_cls: Type = cls._registry[name]
        return model_cls(**kwargs)

    @classmethod
    def list_models(cls) -> KeysView[str]:
        """
        Get all registered model names.

        Returns
        -------
        KeysView[str]
            Iterable view of registered model names.
        """
        return cls._registry.keys()


