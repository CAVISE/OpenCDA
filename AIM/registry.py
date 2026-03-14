import inspect
import os
from typing import Dict
import torch


class ModelRegistry:
    _model_registry: dict[str, type] = {}
    _model_wrapper_registry: dict[str, type] = {}

    @classmethod
    def register_model(cls, model_cls):
        name = model_cls.__name__
        if name in cls._model_registry:
            raise ValueError(f"Duplicate model name: {name}")

        cls._model_registry[name] = model_cls

    @classmethod
    def register_model_wrapper(cls, model_wrapper_cls):
        name = model_wrapper_cls.__name__
        if name in cls._model_wrapper_registry:
            raise ValueError(f"Duplicate model wrapper name: {name}")

        cls._model_wrapper_registry[name] = model_wrapper_cls

    @classmethod
    def get_model_by_name(cls, model_name: str):
        if model_name not in cls._model_registry:
            raise KeyError(f"Unknown model '{model_name}'. Available: {list(cls._model_registry)}")

        return cls._model_registry[model_name]

    @classmethod
    def get_model_wrapper(cls, model_wrapper_name: str, **aim_config: Dict):
        if model_wrapper_name not in cls._model_wrapper_registry:
            raise KeyError(f"Unknown model wrapper '{model_wrapper_name}'. Available: {list(cls._model_wrapper_registry)}")

        model_wrapper_cls = cls._model_wrapper_registry[model_wrapper_name]
        model_name = aim_config.pop("model")
        model_cls = cls.get_model_by_name(model_name)

        if model_cls.model_wrapper != model_wrapper_name:
            raise KeyError(f"Incorrect model '{model_name}' for '{model_wrapper_name}'. Available models: {list(cls._model_registry)}")

        model_dir = cls.get_model_location_path(model_name)
        weights = aim_config.pop("weights")
        weights_path = os.path.join(model_dir, "weights", weights)

        if not os.path.exists(weights_path):
            raise Exception(f"No such checkpoints file path found: {weights_path}")

        model_params = aim_config.pop("model_params")
        model = model_cls(**model_params)

        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

        # remove 'models_list.0.' prefix if present (becuse erlier while saving state_dict with model_factory there was wrap around model)
        if state_dict and any(key.startswith("models_list.0.") for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("models_list.0."):
                    new_key = key[len("models_list.0.") :]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict)

        return model_wrapper_cls(model, **aim_config)

    @classmethod
    def get_model_location_path(cls, name):
        if name in cls._model_registry:
            model_cls = cls._model_registry[name]
        else:
            raise KeyError(f"Unknown model '{name}'. Available: {list(cls._model_registry)}")
        return os.path.dirname(inspect.getfile(model_cls))

    @classmethod
    def list_model_wrappers(cls):
        return cls._model_wrapper_registry.keys()

    @classmethod
    def list_models(cls):
        return cls._model_registry.keys()
