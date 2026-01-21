import inspect


class ModelRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, model_cls):
        if inspect.isabstract(model_cls):
            return

        name = model_cls.__name__
        if name in cls._registry:
            raise ValueError(f"Duplicate model name: {name}")

        cls._registry[name] = model_cls

    @classmethod
    def get_model(cls, name, *args, **kwargs):
        if name in cls._registry:
            model_cls = cls._registry[name]
        else:
            raise KeyError(f"Unknown model '{name}'. Available: {list(cls._registry)}")
        return model_cls(*args, **kwargs)

    @classmethod
    def list_models(cls):
        return cls._registry.keys()
