import argparse
import logging
import os
import shutil


from .data_path_config import DATA_PATH, SUMO_GENED_MAPS_PATH, START_POSITIONS_FILE, LAST_POSITIONS_FILE
from .learning_src.data_scripts.data_config import (
    DT,
    OBS_LEN,
    NUM_PREDICT,
    SAMPLE_RATE,
    VEHICLE_MAX_SPEED,
    VEHICLE_ACCEL,
    VEHICLE_DECEL,
    VEHICLE_SIGMA,
    VEHICLE_LENGTH,
    VEHICLE_MIN_GAP,
)
from .learning_src.data_scripts.generate_csv_utils import (
    generate_csv_from_fcd,
    generate_fcd,
    generate_routefile,
    generate_sumocfg,
    get_map_bounding,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--num_seconds", type=int, help="", default=1000)
    parser.add_argument("--create_new_vehicle_prob", type=float, help="", default=0.08)
    parser.add_argument("--split", type=str, help="train, val or test", default="train")
    parser.add_argument("--random_seed", type=int, help="", default=7)
    parser.add_argument(
        "--nets_dir",
        type=str,
        help="Path to all sumo maps, must locate in CoDriving/data/",
        default="sumo_maps",
    )

    args = parser.parse_args()

    num_seconds = args.num_seconds
    create_new_vehicle_prob = args.create_new_vehicle_prob
    split = args.split
    random_seed = args.random_seed

    nets_dir = args.nets_dir
    nets_dir_path = os.path.join(DATA_PATH, nets_dir)

    if os.path.exists(SUMO_GENED_MAPS_PATH):  # delete directory with old data if exists
        shutil.rmtree(SUMO_GENED_MAPS_PATH)

    TRAFFIC_SCALE = 1.0  # regulate the traffic flow
    LENGTH_PER_SCENE = (NUM_PREDICT + OBS_LEN) // SAMPLE_RATE

    for net_file_name in os.listdir(nets_dir_path):
        net_name = net_file_name.split(sep=".")[0]
        src_net_file_path = os.path.join(nets_dir_path, net_file_name)

        sumo_files_path = os.path.join(SUMO_GENED_MAPS_PATH, net_name)
        route_filename = f"{net_name}-{num_seconds:0>5}-{create_new_vehicle_prob}-{split}-{random_seed}"
        dest_map_dir_path = os.path.join(sumo_files_path, "map")

        os.makedirs(dest_map_dir_path, exist_ok=True)
        shutil.copy(src_net_file_path, os.path.join(dest_map_dir_path, net_file_name))
        shutil.copytree(os.path.join(DATA_PATH, "base_sumocfg"), os.path.join(sumo_files_path, "sumocfg"))

        logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
        map_bounding = get_map_bounding(src_net_file_path)

        # Generate rou.xml and sumocfg files
        generate_routefile(
            sumo_files_path=sumo_files_path,
            rou_xml_filename=route_filename,
            net_xml_filename=net_file_name,
            num_seconds=num_seconds,
            create_new_vehicle_prob=create_new_vehicle_prob,
            random_seed=random_seed,
            max_vehicle_accel=VEHICLE_ACCEL * map_bounding,
            vehicle_decel=VEHICLE_DECEL * map_bounding,
            vehicle_sigma=VEHICLE_SIGMA,
            vehicle_length=VEHICLE_LENGTH * map_bounding,
            vehicle_max_speed=VEHICLE_MAX_SPEED * map_bounding,
            vehicle_minGap=VEHICLE_MIN_GAP * map_bounding,
        )
        sumocfg_path = generate_sumocfg(sumo_files_path, route_filename, net_file_name)

        fcd_dir_path = os.path.join(SUMO_GENED_MAPS_PATH, net_name, "fcd")
        csv_dir_path = os.path.join(DATA_PATH, "csv", split, net_name)
        for dir in [csv_dir_path, fcd_dir_path]:
            os.makedirs(dir, exist_ok=True)

        # Generate a fcd file
        fcd_file = os.path.join(fcd_dir_path, f"{route_filename}.xml")
        logging.info(f"Generating {fcd_file}...")
        generate_fcd(sumocfg_path, fcd_file, 0, 0, num_seconds, DT, TRAFFIC_SCALE)

        # Generate csv files
        logging.info(f"Generating csv files in csv/{split}...")
        generate_csv_from_fcd(fcd_file, csv_dir_path, LENGTH_PER_SCENE, map_bounding, START_POSITIONS_FILE, LAST_POSITIONS_FILE)
