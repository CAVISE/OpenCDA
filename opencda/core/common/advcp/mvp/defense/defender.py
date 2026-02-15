from typing import Any


class Defender:
    def __init__(self) -> None:
        self.name = "base"

    def run(self, multi_frame_case: Any, defend_opts: Any) -> Any:
        raise NotImplementedError
