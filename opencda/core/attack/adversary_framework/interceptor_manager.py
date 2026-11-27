"""Independent, ordered output interceptors for synchronous service handlers."""

from collections.abc import Callable
from functools import wraps
from typing import Any

from .models.interceptor_binding import InterceptorBinding


class InterceptorManager:
    """Manage output rewrites on synchronous service methods.

    Each service instance and method name share one dispatcher. It calls the
    original handler once, then applies active rewrites in registration order.
    Removing the final rewrite restores the original attribute lookup unless
    another owner has already replaced the dispatcher.
    """

    def __init__(self) -> None:
        """Start with no registered service methods."""
        self._bindings: dict[tuple[int, str], InterceptorBinding] = {}

    def install(self, service: Any, binding: Callable[..., Any], rewrite: Callable[[Any], Any]) -> Callable[[], None]:
        """Register a rewrite for calls to a service method.

        Args:
            service: Instance exposing the method to wrap.
            binding: Named callable implementing the service method.
            rewrite: Function receiving the preceding result and returning
                the value to pass to the next rewrite or the caller.

        Returns:
            A callback that removes this registration. Repeated calls are safe.

        Raises:
            RuntimeError: The binding has no name, or the registered dispatcher
                has been replaced on the service instance.
        """
        name = getattr(binding, "__name__", None)
        if not name:
            raise RuntimeError("Capability handler must expose a method name.")
        key = (id(service), name)
        state = self._bindings.get(key)
        if state is None:
            overrides = vars(service)
            callbacks: dict[object, Callable[[Any], Any]] = {}

            @wraps(binding)
            def dispatch(*args: Any, **kwargs: Any) -> Any:
                """Call the handler and apply rewrites still active at their turn."""
                result = binding(*args, **kwargs)
                # A rewrite may remove another registration during dispatch.
                for token, callback in tuple(callbacks.items()):
                    if token in callbacks:
                        result = callback(result)
                return result

            state = InterceptorBinding(dispatch=dispatch, overridden=name in overrides, previous=overrides.get(name), callbacks=callbacks)
            setattr(service, name, dispatch)
            self._bindings[key] = state
        elif getattr(service, name) is not state.dispatch:
            raise RuntimeError(f"Service handler '{name}' was replaced outside the interceptor manager.")

        token = object()
        state.callbacks[token] = rewrite

        def remove() -> None:
            """Release this registration and restore the method after the last removal."""
            if token not in state.callbacks:
                return
            del state.callbacks[token]
            if state.callbacks:
                return
            if vars(service).get(name) is state.dispatch:
                if state.overridden:
                    setattr(service, name, state.previous)
                else:
                    delattr(service, name)
            self._bindings.pop(key, None)

        return remove


interceptor_manager = InterceptorManager()
