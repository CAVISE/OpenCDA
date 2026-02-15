from typing import Any


class Dataset:
    def __init__(self, name: str = "") -> None:
        self.name = name

    @staticmethod
    def load_feature(case: Any, feature_data: Any) -> Any:
        for frame_id, frame_data in enumerate(feature_data):
            for vehicle_id, vehicle_data in frame_data.items():
                case[frame_id][vehicle_id].update(vehicle_data)

        return case
