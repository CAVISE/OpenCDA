from typing import Any


class Perception:
    def __init__(self) -> None:
        pass

    def run(self, case: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def run_single_vehicle(self, case: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def run_multi_vehicle(self, case: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def run_multi_frame(self, case: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()
