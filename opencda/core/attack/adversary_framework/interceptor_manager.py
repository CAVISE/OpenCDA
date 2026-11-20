"""Independent, ordered output interceptors for synchronous service handlers."""

from collections.abc import Callable
from functools import wraps
from typing import Any


class InterceptorManager:
    """Keep one dispatcher per service method; remove registrations in any order."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[int, str], dict[str, Any]] = {}

    def install(self, service: Any, binding: Callable[..., Any], rewrite: Callable[[Any], Any]) -> Callable[[], None]:
        """Register a rewrite and return an idempotent removal handle."""
        name = getattr(binding, "__name__", None)
        if not name:
            raise RuntimeError("Capability handler must expose a method name.")
        key = (id(service), name)
        state = self._bindings.get(key)
        if state is None:
            overrides = vars(service)
            state = {"callbacks": {}, "overridden": name in overrides, "previous": overrides.get(name)}

            @wraps(binding)
            def dispatch(*args: Any, **kwargs: Any) -> Any:
                result = binding(*args, **kwargs)
                # A rewrite may remove another registration during dispatch.
                for token, callback in tuple(state["callbacks"].items()):
                    if token in state["callbacks"]:
                        result = callback(result)
                return result

            state["dispatch"] = dispatch
            setattr(service, name, dispatch)
            self._bindings[key] = state
        elif getattr(service, name) is not state["dispatch"]:
            raise RuntimeError(f"Service handler '{name}' was replaced outside the interceptor manager.")

        token = object()
        state["callbacks"][token] = rewrite

        def remove() -> None:
            if token not in state["callbacks"]:
                return
            del state["callbacks"][token]
            if state["callbacks"]:
                return
            if vars(service).get(name) is state["dispatch"]:
                if state["overridden"]:
                    setattr(service, name, state["previous"])
                else:
                    delattr(service, name)
            self._bindings.pop(key, None)

        return remove


interceptor_manager = InterceptorManager()
