import os
import re
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger("cavise.opencda.opencda.core.common.directory_processor")


class DirectoryProcessor:
    def __init__(self, source_directory: str = "data_dumping", max_cav: Optional[int] = None) -> None:
        self.source_directory = source_directory
        self.max_cav = int(max_cav) if max_cav is not None else None

    def detect_cameras(self, data_directory: str) -> list[str]:
        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])
        if not inner_subdirectories:
            return []

        sample_folder = os.path.join(data_directory, inner_subdirectories[0])
        camera_files = [f for f in os.listdir(sample_folder) if re.match(r"\d+_camera\d+\.png", f)]

        camera_ids = sorted(set(re.findall(r"_camera(\d+)\.png", f)[0] for f in camera_files if re.findall(r"_camera(\d+)\.png", f)))

        return [f"_camera{cam_id}.png" for cam_id in camera_ids]

    def retrieve_data_structure(self, tick_number: int) -> Optional[OrderedDict[int, OrderedDict[str, object]]]:
        number = f"{tick_number:06d}"

        subdirectories = sorted([d for d in os.listdir(self.source_directory) if os.path.isdir(os.path.join(self.source_directory, d))])

        if len(subdirectories) < 2:
            return None

        data_directory = os.path.join(self.source_directory, subdirectories[-2])

        try:
            camera_postfixes = self.detect_cameras(data_directory)
        except Exception:
            camera_postfixes = []

        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])

        if not inner_subdirectories:
            return None

        if "rsu" in inner_subdirectories[0]:
            inner_subdirectories = inner_subdirectories[1:] + [inner_subdirectories[0]]

        if self.max_cav is not None and self.max_cav > 0 and len(inner_subdirectories) > self.max_cav:
            logger.warning(f"Too many CAVs and RSUs: {len(inner_subdirectories)}")
            logger.warning(f"Maximum is {self.max_cav}")
            inner_subdirectories = inner_subdirectories[: self.max_cav]

        expected_ego_id = inner_subdirectories[0]
        expected_agent_path = os.path.join(data_directory, expected_ego_id)
        expected_yaml_path = os.path.join(expected_agent_path, f"{number}.yaml")
        expected_lidar_path = os.path.join(expected_agent_path, f"{number}.pcd")
        if not os.path.exists(expected_yaml_path) or not os.path.exists(expected_lidar_path):
            logger.warning(f"Skipping tick {tick_number}: expected ego agent '{expected_ego_id}' has incomplete data.")
            return None

        scenario_data: OrderedDict[int, OrderedDict[str, object]] = OrderedDict()
        scenario_data[0] = OrderedDict()

        agents_found_count = 0

        for j, folder in enumerate(inner_subdirectories):
            cav_id = folder
            agent_path = os.path.join(data_directory, cav_id)

            yaml_path = os.path.join(agent_path, f"{number}.yaml")
            lidar_path = os.path.join(agent_path, f"{number}.pcd")

            if not os.path.exists(yaml_path) or not os.path.exists(lidar_path):
                continue

            agent_record: OrderedDict[str, object] = OrderedDict()
            scenario_data[0][cav_id] = agent_record
            timestamp = number
            snapshot_record: OrderedDict[str, object] = OrderedDict()
            agent_record[timestamp] = snapshot_record

            snapshot_record["yaml"] = yaml_path
            snapshot_record["lidar"] = lidar_path

            camera_files = []
            for postfix in camera_postfixes:
                cam_path = os.path.join(agent_path, f"{number}{postfix}")

                if os.path.exists(cam_path):
                    camera_files.append(cam_path)
                else:
                    pass

            snapshot_record["camera0"] = camera_files

            agent_record["ego"] = cav_id == expected_ego_id

            agents_found_count += 1

        if agents_found_count == 0:
            return None

        return scenario_data
