import pkgutil
import importlib


# Init all subdirectories as modules
def _discover_model_packages() -> None:
    for info in pkgutil.iter_modules(__path__, prefix="AIM.models."):
        if not info.ispkg:
            continue

        importlib.import_module(info.name)


_discover_model_packages()
