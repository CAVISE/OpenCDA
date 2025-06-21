import os
import random
import shutil
import xml.dom.minidom
from typing import NoReturn

import numpy as np
import pandas as pd
import sumolib
from tqdm import trange
from utils.config import COLLECT_DATA_RADIUS, OBS_LEN, PRED_LEN, SAMPLE_RATE
from utils.feature_utils import get_path_to_intention, get_center_coodinates


def get_shortest_path(net_path: str, from_edge: str, to_edge: str) -> str:
    net = sumolib.net.readNet(net_path)
    from_edge = net.getEdge(from_edge)
    to_edge = net.getEdge(to_edge)

    edges_short, _ = net.getShortestPath(from_edge, to_edge)

    if edges_short is None:
        return ""

    return " ".join([i.getID() for i in edges_short])


def save_csv(df: pd.DataFrame, name: str, dir: str = "./csv") -> NoReturn:
    if not os.path.exists(dir):
        os.makedirs(dir)

    csv_name = f"{name}.csv"
    df.to_csv(os.path.join(dir, csv_name))


def generate_fcd(
    sumocfg_path: str,
    fcd_path: str,
    begin_time: int,
    offset_time: int,
    total_time: int,
    step_length: float = 0.1,
    traffic_scale: float = 0.5,
) -> NoReturn:
    cmd = f"sumo -c {sumocfg_path} --fcd-output {fcd_path} --begin {begin_time} --end {begin_time+offset_time+total_time} \
            --step-length {step_length} --scale {traffic_scale}"
    os.system(cmd)


def get_random_route(from_path: str, possible_paths: dict[str, list[str]]) -> str:
    """
    Args:
        - from_path: e.g. from_edge id

    Return:
        - route name: e.g. '<from_edge>_<to_edge>'
    """
    return f"{from_path}_{random.choice(list(possible_paths[from_path]))}"


def generate_routefile(
    sumo_files_path: str,
    rou_xml_filename: str,
    net_xml_filename: str = "simple_separate_10m.net.xml",
    intention_config_filename: str = "simple_separate_10m_intentions.json",
    num_seconds: int = 2000,
    create_new_vehicle_prob: float = 0.08,
    random_seed: int = 3,
) -> NoReturn:
    """
    Generate *.rou.xml file. (for the separated road net)

    Args:
        - sumo_files_path
        - rou_xml_filename
        - net_xml_filename
        - from_edges: from and to_edges use to generate all possible routes between them
        - to_edges
        - num_seconds
        - create_new_vehicle_prob: the prob of generating a new vehicle at a start point per second, e.g. 0.08 (normal), 0.12 (slightly busy)
        - straight_prob
        - random_seed
    """
    random.seed(random_seed)  # make tests reproducible
    num_vehicles = 0
    net_xml_path = os.path.join(sumo_files_path, "map", net_xml_filename)
    intention_config_path = os.path.join(sumo_files_path, "intentions", intention_config_filename)
    route_path = os.path.join(sumo_files_path, "route")
    os.makedirs(route_path, exist_ok=True)

    file_path = os.path.join(route_path, f"{rou_xml_filename}.rou.xml")
    with open(file_path, "w") as route_file:
        route_file.write("""<routes>
    <vType id="typeWE" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="40" guiShape="passenger"/>\n\n""")

        possible_paths = {}
        paths = get_path_to_intention(intention_config_path).keys()
        for path in paths:
            from_edge, to_edge = path.split("_")
            route = get_shortest_path(net_xml_path, from_edge, to_edge)
            if route == "":
                raise Exception(f"There is no path between {from_edge} and {to_edge}")
            route_file.write(f'    <route id="{from_edge}_{to_edge}" edges="{route}"/>\n')
            if from_edge not in possible_paths:
                possible_paths[from_edge] = []
            possible_paths[from_edge].append(to_edge)

        for second in range(num_seconds):
            for from_edge in possible_paths:
                random_value = random.uniform(0, 1)
                if random_value < create_new_vehicle_prob:
                    route = get_random_route(from_edge, possible_paths)
                    route_file.write(f'    <vehicle id="{route}_{num_vehicles}" type="typeWE" route="{route}" depart="{second}" />\n')
                    num_vehicles += 1

        route_file.write("</routes>")


def generate_sumocfg(sumo_files_path: str, rou_xml_filename: str, net_filename: str) -> str:
    sumocfg_path = os.path.join(sumo_files_path, "sumocfg")
    os.makedirs(sumocfg_path, exist_ok=True)
    sumocfg_filename = os.path.join(sumocfg_path, f"{rou_xml_filename}.sumocfg")

    template_path = os.path.join(sumocfg_path, "template.sumocfg")
    with open(template_path, "r") as f:
        template = f.read()

    filled_cfg = template.format(net=net_filename, rou=rou_xml_filename)

    with open(sumocfg_filename, "w") as out:
        out.write(filled_cfg)
    return sumocfg_filename


def generate_csv_from_fcd(fcd_file: str, intention_config_path: str, time_per_scene: int, split: str = "train"):
    csv_dir = os.path.join("csv", split)
    if os.path.exists(csv_dir):  # delete directory with old data if exists
        shutil.rmtree(csv_dir)

    DOMTree = xml.dom.minidom.parse(fcd_file)
    collection = DOMTree.documentElement
    tracks = collection.getElementsByTagName("timestep")
    df = pd.DataFrame()
    tgt_agent_ids = []
    center_coordinates = get_center_coodinates(intention_config_path)

    for t in trange(len(tracks)):  # each timestamp (0.1s)
        track = tracks[t]
        timestamp = float(track.getAttribute("time"))
        vehicles = track.getElementsByTagName("vehicle")
        # add cars from current timestamp
        for vehicle in vehicles:
            track_id = vehicle.getAttribute("id")
            x = float(vehicle.getAttribute("x"))
            y = float(vehicle.getAttribute("y"))
            yaw_angle = float(vehicle.getAttribute("angle"))
            speed = float(vehicle.getAttribute("speed"))
            new_row = pd.DataFrame(
                [
                    {
                        "TIMESTAMP": timestamp,
                        "TRACK_ID": track_id,
                        "OBJECT_TYPE": "tgt",
                        "X": x,
                        "Y": y,
                        "yaw": yaw_angle,
                        "speed": speed,
                        "CITY_NAME": "SUMO",
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
        if len(df) == 0:
            continue

        curr_time = df["TIMESTAMP"].max() - (PRED_LEN / SAMPLE_RATE)
        max_time = int(df["TIMESTAMP"].max())
        min_time = max_time - time_per_scene

        for track_id, remain_df in df.groupby("TRACK_ID"):
            nearby_data = remain_df.loc[np.isclose(remain_df["TIMESTAMP"], curr_time)]
            if len(nearby_data) == 0:
                continue
            if len(remain_df) < PRED_LEN + OBS_LEN:
                continue
            x, y = nearby_data[["X", "Y"]].values.reshape(-1)
            if (-COLLECT_DATA_RADIUS < x - center_coordinates["x"] < COLLECT_DATA_RADIUS) and (
                -COLLECT_DATA_RADIUS < y - center_coordinates["y"] < COLLECT_DATA_RADIUS
            ):
                tgt_agent_ids.append(track_id)

        if len(tgt_agent_ids) > 0:
            df = df.drop(df[df.TIMESTAMP < (df["TIMESTAMP"].max() - time_per_scene)].index)  # make sure each scene is exactly time_per_scene length
            csv_df = df.loc[[id in tgt_agent_ids for id in df["TRACK_ID"].values.tolist()]]
            csv_name = f"{(min_time):0>5}-{(max_time):0>5}"
            save_csv(csv_df, csv_name, csv_dir)
            df = df.drop(
                df[df.TIMESTAMP <= (df["TIMESTAMP"].max() - time_per_scene + 3.5)].index
            )  # sliding window of 3.5 seconds (avoid overlap between 2 csv)
            del csv_df
            tgt_agent_ids = []
