import os
import random
import shutil
import xml.dom.minidom
from typing import NoReturn
import numpy as np
import pandas as pd
import sumolib
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from tqdm import trange

from .data_config import (
    COLLECT_DATA_RADIUS,
    OBS_LEN,
    NUM_PREDICT,
    SAMPLE_RATE,
)


def get_shortest_path(net_path: str, from_edge: str, to_edge: str) -> str:
    """
    get shortest path from one edge to another

    :param net_path: path to net.xml file
    :param from_edge: start edge id
    :param to_edge: end edge id

    :return path: path in edge sequence between from_edge and to_edge
    """
    net = sumolib.net.readNet(net_path)
    from_edge = net.getEdge(from_edge)
    to_edge = net.getEdge(to_edge)

    edges_short, _ = net.getShortestPath(from_edge, to_edge)

    if edges_short is None:
        return ""

    return " ".join([i.getID() for i in edges_short])


def get_map_bounding(src_net_file_path: str) -> float:
    """
    get map bounding from net_file

    :param src_net_file_path: path to net.xml file

    :return map boundings: one number (expected that map is represented as squere)
    """
    if os.path.exists(src_net_file_path) and src_net_file_path.endswith(".net.xml"):
        tree = ET.parse(src_net_file_path)
        root = tree.getroot()

        map_config = root.find("location")
        x_min, y_min, x_max, y_max = list(map(float, map_config.get("convBoundary").split(sep=",")))
        assert abs(x_min) == abs(y_min) == abs(x_max) == abs(y_max)
        return x_max

    else:
        raise Exception("")


def get_entry_exit_edges(net_path: str) -> dict[str, list[str]]:
    """
    Gets all entry edges and for each finds all possible exit edges that a vehicle can reach.

    Entry edges are "boundary" edges that have only one connection (no incoming edges or with minimal count).
    Exit edges are "boundary" edges that have only one connection (no outgoing edges or with minimal count).

    :param net_path: path to .net.xml file

    :return dictionary: dictionary where keys are entry edge IDs and values are lists of exit edge IDs that can be reached from the corresponding entry edge
    """
    net = sumolib.net.readNet(net_path)
    entry_edges = []
    exit_edges = []

    all_edges = [e for e in net.getEdges() if e.getPriority() >= 0 and not e.getID().startswith(":")]

    if not all_edges:
        all_edges = [e for e in net.getEdges() if not e.getID().startswith(":")]

    min_incoming = min(len(e.getIncoming()) for e in all_edges) if all_edges else 0
    min_outgoing = min(len(e.getOutgoing()) for e in all_edges) if all_edges else 0

    entry_edges = [e for e in all_edges if len(e.getIncoming()) == min_incoming]
    exit_edges = [e for e in all_edges if len(e.getOutgoing()) == min_outgoing]

    result = {}
    for entry_edge in entry_edges:
        entry_id = entry_edge.getID()
        reachable_exits = []

        for exit_edge in exit_edges:
            exit_id = exit_edge.getID()

            if entry_id == exit_id:
                continue

            path, _ = net.getShortestPath(entry_edge, exit_edge)
            if path is not None:
                reachable_exits.append(exit_id)

        if reachable_exits:
            result[entry_id] = reachable_exits

    return result


def get_connection_via(root: Element, from_edge: str, to_edge: str) -> str:
    """
    Find the internal edge (via) that connects two edges.

    :param root: root elemet of net.xml file
    :param from_edge: start edge
    :param to_edge: end egde
    """
    for connection in root.findall("connection"):
        conn_from = connection.get("from")
        conn_to = connection.get("to")

        if conn_from == from_edge and conn_to == to_edge and not conn_from.startswith(":"):
            via = connection.get("via")
            if via:
                via_edge = via.rsplit("_", 1)[0]
                return via_edge
    return None


def save_csv(df: pd.DataFrame, name: str, dir: str = "./csv") -> NoReturn:
    """
    save dataframe to csv

    :param df: dataframe
    :param name: name of csv file to save to
    :param dir: path to dir to place csv file to
    """
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
    precision: int = 6,
    is_ballistic_mode: bool = False,
) -> NoReturn:
    """
    generating fcd from sumoconfig file

    :param sumocfg_path: path to sumo config file
    :param fcd_path: path to output fcd file
    :param begin_time: begin time of simulation
    :param offset_time: time offset before starting data collection
    :param total_time: total time of simulation
    :param step_length: simulation step length in seconds
    :param traffic_scale: traffic scale factor
    :param precision: decimal precision for output coordinates
    :param is_ballistic_mode: flag if ballistic mode
    """

    cmd = f"sumo -c {sumocfg_path} --fcd-output {fcd_path} --begin {begin_time} --end {begin_time + offset_time + total_time} \
            --step-length {step_length} --scale {traffic_scale} --precision {precision}"

    if is_ballistic_mode:
        cmd = f"{cmd} --step-method.ballistic"

    os.system(cmd)


def get_random_route(from_path: str, possible_paths: dict[str, list[str]]) -> str:
    """
    get random route from possible paths

    :param from_path: start edge id (e.g. from_edge id)
    :param possible_paths: dictionary mapping entry edges to list of exit edges

    :return: route name in format '<from_edge>_<to_edge>'
    """
    return f"{from_path}_{random.choice(list(possible_paths[from_path]))}"


def generate_routefile(
    sumo_files_path: str,
    rou_xml_filename: str,
    net_xml_filename: str,
    num_seconds: int = 2000,
    create_new_vehicle_prob: float = 0.08,
    random_seed: int = 3,
    max_vehicle_accel: float = 0.025,
    vehicle_decel: float = 0.045,
    vehicle_sigma: float = 0.5,
    vehicle_length: float = 0.05,
    vehicle_max_speed: float = 0.4,
    vehicle_minGap: float = 0.025,
) -> NoReturn:
    """
    generate route file (*.rou.xml) for separated road network

    :param sumo_files_path: path to directory containing sumo files
    :param rou_xml_filename: name of output route xml file (without extension)
    :param net_xml_filename: name of network xml file
    :param num_seconds: duration of simulation in seconds
    :param create_new_vehicle_prob: probability of creating new vehicle at each second
    :param random_seed: random seed for reproducibility
    :param max_vehicle_accel: maximum vehicle acceleration
    :param vehicle_decel: vehicle deceleration rate
    :param vehicle_sigma: driver imperfection parameter
    :param vehicle_length: vehicle length
    :param vehicle_max_speed: maximum vehicle speed
    :param vehicle_minGap: minimum gap between vehicles
    """
    random.seed(random_seed)  # make tests reproducible
    num_vehicles = 0
    net_xml_path = os.path.join(sumo_files_path, "map", net_xml_filename)
    route_path = os.path.join(sumo_files_path, "route")
    os.makedirs(route_path, exist_ok=True)

    file_path = os.path.join(route_path, f"{rou_xml_filename}.rou.xml")
    with open(file_path, "w") as route_file:
        route_file.write(
            f"""<routes>
    <vType id="typeWE" accel="{max_vehicle_accel}" decel="{vehicle_decel}" sigma="{vehicle_sigma}" length="{vehicle_length}" minGap="{vehicle_minGap}" maxSpeed="{vehicle_max_speed}" guiShape="passenger"/>\n\n"""
        )

        possible_paths = get_entry_exit_edges(net_path=net_xml_path)

        for start_edge, end_edges in possible_paths.items():
            for end_edge in end_edges:
                route = get_shortest_path(net_xml_path, start_edge, end_edge)
                if route == "":
                    raise Exception(f"There is no path between {start_edge} and {end_edge}")

                route_file.write(f'    <route id="{start_edge}_{end_edge}" edges="{route}"/>\n')

        for second in range(num_seconds):
            for from_edge in possible_paths:
                random_value = random.uniform(0, 1)
                if random_value < create_new_vehicle_prob:
                    route = get_random_route(from_edge, possible_paths)
                    route_file.write(f'    <vehicle id="{route}_{num_vehicles}" type="typeWE" route="{route}" depart="{second}" />\n')
                    num_vehicles += 1

        route_file.write("</routes>")


def generate_sumocfg(sumo_files_path: str, rou_xml_filename: str, net_filename: str) -> str:
    """
    generate sumo config file from template

    :param sumo_files_path: path to directory containing sumo files
    :param rou_xml_filename: name of route xml file
    :param net_filename: name of network xml file

    :return: path to generated sumo config file
    """
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


def generate_csv_from_fcd(
    fcd_file: str, csv_dir: str, time_per_scene: int, map_boundings: float, start_positions_file: str, last_positions_file: str
) -> NoReturn:
    """
    generate csv files from sumo fcd (floating car data) file

    :param fcd_file: path to input fcd xml file
    :param csv_dir: directory to save output csv files
    :param time_per_scene: duration of each scene in seconds
    :param map_boundings: map boundary size
    :param start_positions_file: filename for saving initial vehicle positions
    :param last_positions_file: filename for saving final vehicle positions
    """
    if os.path.exists(csv_dir):  # delete directory with old data if exists
        shutil.rmtree(csv_dir)

    DOMTree = xml.dom.minidom.parse(fcd_file)
    collection = DOMTree.documentElement
    tracks = collection.getElementsByTagName("timestep")
    df = pd.DataFrame()
    tgt_agent_ids = []

    start_cars_info = {}
    last_cars_info = {}
    collect_data_radius = COLLECT_DATA_RADIUS * map_boundings

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

        curr_time = df["TIMESTAMP"].max() - (NUM_PREDICT / SAMPLE_RATE)
        max_time = int(df["TIMESTAMP"].max())
        min_time = max_time - time_per_scene

        for track_id, remain_df in df.groupby("TRACK_ID"):
            nearby_data = remain_df.loc[np.isclose(remain_df["TIMESTAMP"], curr_time)]
            if len(nearby_data) == 0:
                continue

            norm = np.sqrt(remain_df["X"] ** 2 + remain_df["Y"] ** 2)
            mask = norm < collect_data_radius
            norm_remain_df = remain_df.loc[mask, ["TIMESTAMP", "X", "Y", "yaw", "speed"]]
            norm_remain_df = norm_remain_df[norm_remain_df["TIMESTAMP"] >= min_time]

            if len(norm_remain_df) < NUM_PREDICT + OBS_LEN:
                continue

            x, y = nearby_data[["X", "Y"]].values.reshape(-1)
            if (-collect_data_radius < x < collect_data_radius) and (-collect_data_radius < y < collect_data_radius):
                tgt_agent_ids.append(track_id)

            first_row = norm_remain_df.sort_values("TIMESTAMP").iloc[0]
            if track_id not in start_cars_info.keys():
                start_cars_info[track_id] = {
                    "TIMESTAMP": float(first_row["TIMESTAMP"]),
                    "X": float(first_row["X"]),
                    "Y": float(first_row["Y"]),
                    "yaw": float(first_row["yaw"]),
                    "speed": float(first_row["speed"]),
                }

        if len(tgt_agent_ids) > 0:
            df = df.drop(df[df.TIMESTAMP < (df["TIMESTAMP"].max() - time_per_scene)].index)  # make sure each scene is exactly time_per_scene length
            csv_df = df.loc[[id in tgt_agent_ids for id in df["TRACK_ID"].values.tolist()]]

            norm = np.sqrt(csv_df["X"] ** 2 + csv_df["Y"] ** 2)
            mask = norm < collect_data_radius
            csv_df = csv_df.loc[mask]

            for track_id, vehicle_df in csv_df.groupby("TRACK_ID"):
                last_row = vehicle_df.sort_values("TIMESTAMP").iloc[-1]
                last_timestamp = float(last_row["TIMESTAMP"])

                if track_id not in last_cars_info or last_timestamp > last_cars_info[track_id]["TIMESTAMP"]:
                    last_cars_info[track_id] = {
                        "TIMESTAMP": last_timestamp,
                        "X": float(last_row["X"]),
                        "Y": float(last_row["Y"]),
                        "yaw": float(last_row["yaw"]),
                        "speed": float(last_row["speed"]),
                    }

            csv_name = f"{(min_time):0>5}-{(max_time):0>5}"
            save_csv(csv_df, csv_name, csv_dir)
            df = df.drop(
                df[df.TIMESTAMP <= (df["TIMESTAMP"].max() - time_per_scene + 3.5)].index
            )  # sliding window of 3.5 seconds (avoid overlap between 2 csv)
            del csv_df
            tgt_agent_ids = []

    start_cars_df = pd.DataFrame.from_dict(start_cars_info, orient="index")
    start_cars_df.index.name = "TRACK_ID"
    start_cars_df.reset_index(inplace=True)
    start_cars_df.to_csv(os.path.join(csv_dir, start_positions_file))

    last_cars_df = pd.DataFrame.from_dict(last_cars_info, orient="index")
    last_cars_df.index.name = "TRACK_ID"
    last_cars_df.reset_index(inplace=True)
    last_cars_df.to_csv(os.path.join(csv_dir, last_positions_file))
