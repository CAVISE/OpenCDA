import os
import random
import xml.dom.minidom
import shutil

import pandas as pd
import numpy as np
from tqdm import trange

from utils.config import OBS_LEN, PRED_LEN, SAMPLE_RATE, COLLECT_DATA_RADIUS


def save_csv(df: pd.DataFrame, name: str, dir: str = './csv'):
    if not os.path.exists(dir):
        os.makedirs(dir)

    csv_name = f"{name}.csv"
    df.to_csv(os.path.join(dir, csv_name))


def generate_fcd(sumocfg_path: str, 
                fcd_path: str, 
                begin_time: int, 
                offset_time: int, 
                total_time: int, 
                step_length: float = 0.1, 
                traffic_scale: float = 0.5) -> None:
    cmd = f"sumo -c {sumocfg_path} --fcd-output {fcd_path} --begin {begin_time} --end {begin_time+offset_time+total_time} \
            --step-length {step_length} --scale {traffic_scale}"
    os.system(cmd)
    return None


def get_routes(from_path: str, straight_prob: float = 0.5)-> str:
    """
    Args:
        - from_path: e.g. 'left', 'right', 'up', 'down'
        - straight_prob: the probability of a vehicle to go straight

    Return:
        - route name: e.g. 'left_right', 'up_right'...
    """
    left_turn_prob = (1 - straight_prob) / 2.0

    random_value = random.uniform(0, 1)
    if random_value < straight_prob: # straight
        if from_path == 'left':
            return 'left_right'
        elif from_path == 'right':
            return 'right_left'
        elif from_path == 'up':
            return 'up_down'
        elif from_path == 'down':
            return 'down_up'
    elif random_value < straight_prob + left_turn_prob: # left-turn
        if from_path == 'left':
            return 'left_up'
        elif from_path == 'right':
            return 'right_down'
        elif from_path == 'up':
            return 'up_right'
        elif from_path == 'down':
            return 'down_left'
    else: # right-turn
        if from_path == 'left':
            return 'left_down'
        elif from_path == 'right':
            return 'right_up'
        elif from_path == 'up':
            return 'up_left'
        elif from_path == 'down':
            return 'down_right'


def generate_routefile(sumo_files_path: str,
                        rou_xml_filename: str = '04-16-22-01-00800-0.08-val-4',
                        num_seconds: int = 2000,
                        create_new_vehicle_prob: float = 0.08, 
                        straight_prob: float = 0.4,
                        random_seed: int = 3):
    """
    Generate *.rou.xml file. (for the separated road net)

    Args:
        - rou_xml_filename
        - num_seconds
        - create_new_vehicle_prob: the prob of generating a new vehicle at a start point per second, e.g. 0.08 (normal), 0.12 (slightly busy)
        - straight_prob
    """
    random.seed(random_seed)  # make tests reproducible
    num_vehicles = 0
    route_path = os.path.join(sumo_files_path, "route")
    os.makedirs(route_path, exist_ok=True)

    file_path = os.path.join(route_path, f"{rou_xml_filename}.rou.xml")
    with open(file_path, "w") as route_file:
        route_file.write("""<routes>
        <vType id="typeWE" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="40" guiShape="passenger"/>

        <route id="left_up" edges="E4 -E2 -E1 -E5"/>
        <route id="left_right" edges="E4 -E2 -E3 -E6"/>
        <route id="left_down" edges="E4 -E2 E0 -E7"/>

        <route id="right_up" edges="E6 E3 -E1 -E5"/>
        <route id="right_left" edges="E6 E3 E2 -E4"/>
        <route id="right_down" edges="E6 E3 E0 -E7"/>

        <route id="down_up" edges="E7 -E0 -E1 -E5"/>
        <route id="down_left" edges="E7 -E0 E2 -E4"/>
        <route id="down_right" edges="E7 -E0 -E3 -E6"/>

        <route id="up_down" edges="E5 E1 E0 -E7"/>
        <route id="up_left" edges="E5 E1 E2 -E4"/>
        <route id="up_right" edges="E5 E1 -E3 -E6"/>""")

        for i_second in range(num_seconds):
            if i_second % 50 == 0:
                create_new_vehicle_prob = np.random.randint(6000,8500)/100000

            # from left
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob:
                route = get_routes('left', straight_prob=straight_prob)
                route_file.write('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second))
                num_vehicles += 1

            # from right
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob:
                route = get_routes('right', straight_prob=straight_prob)
                route_file.write('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second))
                num_vehicles += 1

            # from up
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob:
                route = get_routes('up', straight_prob=straight_prob)
                route_file.write('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second))
                num_vehicles += 1

            # from down
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob:
                route = get_routes('down', straight_prob=straight_prob)
                route_file.write('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second))
                num_vehicles += 1

        route_file.write("</routes>")


def generate_sumocfg(sumo_files_path, rou_xml_filename: str = '04-16-22-01-00800-0.08-val-4')-> str:
    sumocfg_path = os.path.join(sumo_files_path, "sumocfg")
    os.makedirs(sumocfg_path, exist_ok=True)
    sumocfg_filename = os.path.join(sumocfg_path, f"{rou_xml_filename}.sumocfg")
    with open(sumocfg_filename, "w") as sumocfg:
        sumocfg.write(f"""<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 08/09/21 14:01:24 by Eclipse SUMO sumo Version v1_8_0+1925-6bf04e0fef
-->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="../map/simple_separate_10m.net.xml"/>
        <route-files value="../route/{rou_xml_filename}.rou.xml"/>
    </input>

    <processing>
        <ignore-route-errors value="true"/>
    </processing>

    <routing>
        <device.rerouting.adaptation-steps value="18"/>
        <device.rerouting.adaptation-interval value="10"/>
    </routing>

    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>

    <gui_only>
        <gui-settings-file value="simple.view.xml"/>
    </gui_only>

</configuration>""")

    return sumocfg_filename

def generate_csv_from_fcd(fcd_file: str, time_per_scene: int, split: str = 'train'):
    csv_dir = os.path.join('csv', split)
    if os.path.exists(csv_dir): # delete directory with old data if exists
        shutil.rmtree(csv_dir)

    DOMTree = xml.dom.minidom.parse(fcd_file)
    collection = DOMTree.documentElement
    tracks = collection.getElementsByTagName('timestep')
    df = pd.DataFrame()
    tgt_agent_ids = []

    for t in trange(len(tracks)): # each timestamp (0.1s)
        track = tracks[t]
        timestamp = float(track.getAttribute('time'))
        vehicles = track.getElementsByTagName('vehicle')
        # add cars from current timestamp
        for vehicle in vehicles:
            track_id = vehicle.getAttribute('id')
            x = float(vehicle.getAttribute('x'))
            y = float(vehicle.getAttribute('y'))
            yaw_angle = float(vehicle.getAttribute('angle'))
            speed = float(vehicle.getAttribute('speed'))
            new_row = pd.DataFrame([{'TIMESTAMP': timestamp, 'TRACK_ID': track_id, 'OBJECT_TYPE': 'tgt', 'X': x, 'Y': y,'yaw':yaw_angle,'speed':speed , 'CITY_NAME': 'SUMO'}])
            df = pd.concat([df, new_row], ignore_index=True)
        if len(df) == 0:
            continue

        curr_time = df['TIMESTAMP'].max() - (PRED_LEN / SAMPLE_RATE)
        max_time = int(df['TIMESTAMP'].max())
        min_time = max_time - time_per_scene

        for track_id, remain_df in df.groupby('TRACK_ID'):
            nearby_data = remain_df.loc[np.isclose(remain_df['TIMESTAMP'], curr_time)]
            if len(nearby_data) == 0:
                continue
            if len(remain_df) < PRED_LEN + OBS_LEN:
                continue
            x, y = nearby_data[['X', 'Y']].values.reshape(-1)
            if (-COLLECT_DATA_RADIUS < x < COLLECT_DATA_RADIUS) and (-COLLECT_DATA_RADIUS < y < COLLECT_DATA_RADIUS):
                tgt_agent_ids.append(track_id)

        if len(tgt_agent_ids) > 0:
            df = df.drop(df[df.TIMESTAMP < (df['TIMESTAMP'].max() - time_per_scene)].index) # make sure each scene is exactly time_per_scene length
            csv_df = df.loc[[id in tgt_agent_ids for id in df['TRACK_ID'].values.tolist()]]
            csv_name = f"{(min_time):0>5}-{(max_time):0>5}"
            save_csv(csv_df, csv_name, csv_dir)
            df = df.drop(df[df.TIMESTAMP <= (df['TIMESTAMP'].max() - time_per_scene + 3.5)].index) # sliding window of 3.5 seconds (avoid overlap between 2 csv)
            del csv_df
            tgt_agent_ids = []

    return None
