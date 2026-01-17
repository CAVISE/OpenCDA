# -*- coding: utf-8 -*-

"""
Script to load dumped YAML files and generate prediction/observed trajectories for each vehicle.

This module provides functionality to process simulation data, extract vehicle trajectories,
and generate prediction and observation data for each vehicle in the simulation.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import concurrent

from concurrent.futures import ThreadPoolExecutor
from opencda.scenario_testing.utils.yaml_utils import load_yaml, save_yaml

from typing import List, Dict


def retrieve_future_params(yaml_params: List, index: int, seconds: int = 8) -> List:
    """
    Retrieve YAML parameters for the next n seconds.

    Parameters
    ----------
    yaml_params : list
        List containing all loaded YAML parameters.
    index : int
        Current timestamp index.
    seconds : int, optional
        Number of future seconds to collect. Default is 8.
        Data collection is at 10Hz, so 8 seconds = 80 frames.

    Returns
    -------
    future_params : list
        YAML parameters for the next n seconds.
    """
    start_index = min(index + 1, len(yaml_params) - 1)
    end_index = min(index + seconds * 10 + 1, len(yaml_params) - 1)
    future_params = yaml_params[start_index:end_index]

    return future_params


def retrieve_past_params(yaml_params: List, index: int, seconds: int = 1) -> List:
    """
    Retrieve YAML parameters for the past n seconds.

    Parameters
    ----------
    yaml_params : list
        List containing all loaded YAML parameters.
    index : int
        Current timestamp index.
    seconds : int, optional
        Number of past seconds to collect. Default is 1.
        Data collection is at 10Hz, so 1 second = 10 frames.

    Returns
    -------
    past_params : list
        YAML parameters for the previous n seconds.
    """
    end_index = max(index - 1, 0)
    start_index = max(index - seconds * 10, 0)
    past_params = yaml_params[start_index:end_index]

    return past_params


def extract_trajectories_by_id(object_id: str, yaml_param_list: List) -> List:
    """
    Extract a vehicle's trajectory from YAML parameters.

    Parameters
    ----------
    object_id : str
        Target vehicle ID.
    yaml_param_list : list
        List containing YAML parameters for trajectory extraction.

    Returns
    -------
    trajectories : list of tuple
        Vehicle's trajectory as a list of 7-tuples:
        (x, y, z, roll, pitch, yaw, speed).
        Coordinates are in world frame with bounding box center as reference..
    """
    trajectories = []

    for yaml_param in yaml_param_list:
        vehicles = yaml_param["vehicles"]

        if int(object_id) not in vehicles:
            break

        target_vehicle = vehicles[int(object_id)]

        location = target_vehicle["location"]
        center = target_vehicle["center"]
        rotation = target_vehicle["angle"]
        speed = target_vehicle["speed"]

        # we regard the center of the bbx as the vehicle true location
        trajectory = (location[0] + center[0], location[1] + center[1], location[2] + center[2], rotation[0], rotation[1], rotation[2], speed)
        trajectories.append(trajectory)

    return trajectories


def extract_trajectories_by_file(yaml_params: List, cur_index: int, past_seconds: int = 1, future_seconds: int = 8) -> Dict:
    """
    Extract predictions and observations for all vehicles at the current timestamp.

    Parameters
    ----------
    yaml_params : list
        List of all loaded YAML dictionaries.
    cur_index : int
        Current file index in the sequence.
    past_seconds : int, optional
        Number of seconds to look back for observation trajectory. Default is 1.
    future_seconds : int, optional
        Number of seconds to look ahead for prediction trajectory. Default is 8.

    Returns
    -------
    cur_param : dict
        Updated YAML parameters with added 'predictions' and 'observations'
        fields for each vehicle.
    """
    cur_param = yaml_params[cur_index]

    for vehicle_id, vehicle in cur_param["vehicles"].items():
        future_yaml_params = retrieve_future_params(yaml_params, cur_index, future_seconds)
        predictions = extract_trajectories_by_id(vehicle_id, future_yaml_params)
        cur_param["vehicles"][vehicle_id].update({"predictions": predictions})

        past_yaml_params = retrieve_past_params(yaml_params, cur_index, past_seconds)
        observations = extract_trajectories_by_id(vehicle_id, past_yaml_params)
        cur_param["vehicles"][vehicle_id].update({"observations": observations})

    return cur_param


def generate_prediction_by_scenario(scenario: str, future_seconds: int = 8, past_seconds: int = 1) -> None:
    """
    Generate prediction and observation trajectories for a single scenario.

    Processes all YAML files in a scenario directory, extracts vehicle trajectories,
    and updates the YAML files with prediction and observation data.

    Parameters
    ----------
    scenario : str
        Path to the directory containing scenario data with CAV subdirectories.
    future_seconds : int, optional
        Number of seconds to look ahead for prediction trajectories. Default is 8.
    past_seconds : int, optional
        Number of seconds to look back for observation trajectories. Default is 1.
    """
    cavs = [os.path.join(scenario, x) for x in os.listdir(scenario) if not x.endswith(".yaml")]
    for j, cav in enumerate(cavs):
        yaml_files = sorted([os.path.join(cav, x) for x in os.listdir(cav) if x.endswith(".yaml")])

        # load all dictionarys at one time
        yaml_params = [load_yaml(x) for x in yaml_files]
        for k in range(len(yaml_files)):
            new_param = extract_trajectories_by_file(yaml_params, k, past_seconds, future_seconds)
            save_yaml(new_param, yaml_files[k])


def generate_prediction_yaml(root_dir: str, future_seconds: int = 8, past_seconds: int = 1) -> None:
    """
    Process all scenarios in the root directory to generate prediction YAMLs.

    Uses parallel processing to handle multiple scenarios concurrently.

    Parameters
    ----------
    root_dir : str
        Root directory containing scenario subdirectories.
    future_seconds : int, optional
        Number of seconds to look ahead for prediction. Default is 8.
    past_seconds : int, optional
        Number of seconds to look back for observation. Default is 1.
    """

    scenarios = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_prediction_by_scenario, scenario, future_seconds, past_seconds) for scenario in scenarios]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    """
    Entry point for trajectory prediction generation.
    """
    root_dir = "../simulation_output/data_dumping/"
    generate_prediction_yaml(root_dir)
