import argparse
import logging
import os

# from lxml import etree
from datetime import datetime
# from functools import partial

from data_path_config import DATA_PATH


from CoDriving.data_scripts.data_config.data_config import DT, OBS_LEN, PRED_LEN, SAMPLE_RATE, VEHICLE_MAX_SPEED
from CoDriving.data_scripts.generate_csv_utils import (
    generate_csv_from_fcd,
    generate_fcd,
    generate_routefile,
    generate_sumocfg,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--num_seconds", type=int, help="", default=1000)
    parser.add_argument("--create_new_vehicle_prob", type=float, help="", default=0.05)
    parser.add_argument("--split", type=str, help="train, val or test", default="train")
    parser.add_argument("--random_seed", type=int, help="", default=7)
    parser.add_argument(
        "--net_filename",
        type=str,
        help="Name of file with sumo network. It must be in data/sumo/map/ directory",
        default="simple_separate_10m.net.xml",
    )
    parser.add_argument(
        "--intention_config",
        type=str,
        help="Name of file with routes and intentins. It must be in data/sumo/intentions/ directory",
        default="simple_separate_10m_intentions.json",
    )

    args = parser.parse_args()

    num_seconds = args.num_seconds
    create_new_vehicle_prob = args.create_new_vehicle_prob
    split = args.split
    random_seed = args.random_seed
    net_filename = args.net_filename
    intention_config = args.intention_config

    now = datetime.now().strftime("%m-%d-%H-%M")
    route_filename = f"{now}-{num_seconds:0>5}-{create_new_vehicle_prob}-{split}-{random_seed}"
    TRAFFIC_SCALE = 1.0  # regulate the traffic flow
    LENGTH_PER_SCENE = (PRED_LEN + OBS_LEN) // SAMPLE_RATE

    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
    sumo_files_path = os.path.join(DATA_PATH, "sumo")

    # Generate rou.xml and sumocfg files
    generate_routefile(
        sumo_files_path=sumo_files_path,
        rou_xml_filename=route_filename,
        num_seconds=num_seconds,
        create_new_vehicle_prob=create_new_vehicle_prob,
        random_seed=random_seed,
        net_xml_filename=net_filename,
        intention_config_filename=intention_config,
        vehicle_max_speed=VEHICLE_MAX_SPEED,
    )
    sumocfg_path = generate_sumocfg(sumo_files_path, route_filename, net_filename)  # type: ignore

    csv_dir_path = os.path.join(DATA_PATH, "csv")
    fcd_dir_path = os.path.join(DATA_PATH, "fcd")
    for dir in [csv_dir_path, fcd_dir_path]:
        os.makedirs(dir, exist_ok=True)

    # Generate a fcd file
    fcd_file = os.path.join(fcd_dir_path, f"{route_filename}.xml")
    logging.info(f"Generating {fcd_file}...")
    generate_fcd(sumocfg_path, fcd_file, 0, 0, num_seconds, DT, TRAFFIC_SCALE)
    intention_config_path = os.path.join(sumo_files_path, "intentions", intention_config)

    # x_normalizer = None
    # y_normalizer = None
    # speed_normalizer = None
    # yaw_normalizer = None

    # if args.normalize:
    #     doc = etree.parse(os.path.join(sumo_files_path, "map", net_filename))
    #     memoryElem = doc.find("location")
    #     boundarie_str = memoryElem.get("convBoundary").split(sep=",")
    #     xmin, ymin, xmax, ymax = [float(num) for num in boundarie_str]

    #     x_normalizer = partial(min_max_normalize, mmax=xmax, mmin=xmin)
    #     y_normalizer = partial(min_max_normalize, mmax=ymax, mmin=ymin)
    #     speed_normalizer = partial(min_max_normalize, mmax=args.vehicle_max_speed, mmin=0)
    #     yaw_normalizer = partial(min_max_normalize, mmax=180)

    # Generate csv files
    logging.info(f"Generating csv files in csv/{split}...")
    generate_csv_from_fcd(
        fcd_file,
        csv_dir_path,
        intention_config_path,
        LENGTH_PER_SCENE,
        split,
        # normalize=args.normalize,
        # x_normalizer=x_normalizer,
        # y_normalizer=y_normalizer,
        # yaw_normalizer=yaw_normalizer,
        # speed_normalizer=speed_normalizer,
    )
