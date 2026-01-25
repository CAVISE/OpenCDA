import inspect
from typing import Any, KeysView, Type


class ModelRegistry:
    """
    Registry for managing model classes.

    This class provides a centralized registry for dynamically registering
    and retrieving model classes by name. It prevents duplicate registrations
    and excludes abstract classes from being registered.
    """
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, model_cls: Type[Any]) -> None:
        """
        Register a model class in the registry.

        Abstract classes are ignored. Raises an error if a model with
        the same name is already registered.

        Parameters
        ----------
        model_cls : Type[Any]
            The model class to register.

        Raises
        ------
        ValueError
            If a model with the same name is already registered
        """
        if inspect.isabstract(model_cls):
            return

        name = model_cls.__name__
        if name in cls._registry:
            raise ValueError(f"Duplicate model name: {name}")

        cls._registry[name] = model_cls

    @classmethod
    def get_model(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve and instantiate a registered model by name.

        Parameters
        ----------
        name : str
            Name of the registered model class.
        *args : Any
            Positional arguments to pass to the model constructor.
        **kwargs : Any
            Keyword arguments to pass to the model constructor.

        Returns
        -------
        Any
            An instance of the requested model class.

        Raises
        ------
        KeyError
            If the model name is not found in the registry.
        """
        if name in cls._registry:
            model_cls = cls._registry[name]
        else:
            raise KeyError(f"Unknown model '{name}'. Available: {list(cls._registry)}")
        return model_cls(*args, **kwargs)

    @classmethod
    def list_models(cls) -> KeysView[str]:
        """
        List all registered model names.

        Returns
        -------
        Iterator[str]
            Iterator over registered model names.
        """
        
        return cls._registry.keys()
