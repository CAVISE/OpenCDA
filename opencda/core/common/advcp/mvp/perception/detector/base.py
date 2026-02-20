from typing import Any


class Detector:
    def __init__(self) -> None:
        pass

    def run(self, pointcloud: Any) -> Any:
        raise NotImplementedError
