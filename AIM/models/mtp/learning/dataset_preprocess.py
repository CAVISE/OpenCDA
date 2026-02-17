import argparse
import os
import concurrent.futures
import sys
import pickle as pkl
import numpy as np
import shutil

from .data_path_config import DATA_PATH, SUMO_GENED_MAPS_PATH, Y_X_DISTR_FILE, Y_Y_DISTR_FILE, START_POSITIONS_FILE, LAST_POSITIONS_FILE
from .learning_src.data_scripts.data_config import NORMALIZE_DATA, ZSCORE_NORMALIZE
from .learning_src.data_scripts.preprocess_utils import preprocess_file, z_score_normalize_file
from .learning_src.data_scripts.preprocess_map import preprocess_map
from .learning_src.data_scripts.generate_csv_utils import get_map_bounding


def get_distribution_params(distr_params_dir, preprocess_folder_path):
    y_x, y_y = [], []
    for i, preprocess_subfolder in enumerate(os.listdir(preprocess_folder_path)):
        if preprocess_subfolder.endswith(".pkl"):
            continue

        preprocess_subfolder_path = os.path.join(preprocess_folder_path, preprocess_subfolder)
        files = [i for i in os.listdir(preprocess_subfolder_path) if os.path.splitext(i)[1] == ".pkl"]
        for file in files:
            data = pkl.load(open(os.path.join(preprocess_subfolder_path, file), "rb"))
            y_x.append(data[1][:, 0::6].flatten())
            y_y.append(data[1][:, 1::6].flatten())

    y_x = np.concatenate(y_x)
    y_y = np.concatenate(y_y)
    pkl.dump((y_x.mean(), y_x.std()), open(os.path.join(distr_params_dir, Y_X_DISTR_FILE), "wb"))
    pkl.dump((y_y.mean(), y_y.std()), open(os.path.join(distr_params_dir, Y_Y_DISTR_FILE), "wb"))


if __name__ == "__main__":
    cpu_amount = os.cpu_count()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--csv_folder",
        type=str,
        help="path to the data set (*.csv)",
        default="csv/train",
    )
    parser.add_argument(
        "--pkl_folder",
        type=str,
        help="path to the preprocessed data (*.pkl)",
        default="csv/train_pre",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help=f"amount of processes(max: {cpu_amount})",
        default=cpu_amount,
    )
    args = parser.parse_args()

    csv_folder = args.csv_folder
    csv_folder_path = os.path.join(DATA_PATH, csv_folder)

    maps_to_process = len(os.listdir(csv_folder_path))

    preprocess_folder = args.pkl_folder
    processes = args.processes
    preprocess_folder_path = os.path.join(DATA_PATH, preprocess_folder)

    if os.path.exists(preprocess_folder_path):
        shutil.rmtree(preprocess_folder_path)
    os.makedirs(preprocess_folder_path, exist_ok=True)

    print("Processing started")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=processes)
    try:
        for i, csv_subfolder in enumerate(os.listdir(csv_folder_path)):
            csv_subfolder_path = os.path.join(csv_folder_path, csv_subfolder)
            preprocess_folder_subpath_path = os.path.join(preprocess_folder_path, csv_subfolder)
            os.makedirs(preprocess_folder_subpath_path, exist_ok=True)

            csv_files = [i for i in os.listdir(csv_subfolder_path) if os.path.splitext(i)[1] == ".csv"]
            net_file_path = os.path.join(SUMO_GENED_MAPS_PATH, csv_subfolder, "map", f"{csv_subfolder}.net.xml")

            map_boundary = get_map_bounding(net_file_path)

            futures = [
                executor.submit(
                    preprocess_file, csv_subfolder_path, file, preprocess_folder_subpath_path, map_boundary, START_POSITIONS_FILE, LAST_POSITIONS_FILE
                )
                for file in csv_files
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")

            preprocess_map(net_file_path=net_file_path, output_dir=preprocess_folder_subpath_path)
            print(f"Progress: {i + 1}/{maps_to_process}")

        if NORMALIZE_DATA and ZSCORE_NORMALIZE:
            dist_params_dir = os.path.join(DATA_PATH, preprocess_folder.split(sep="/")[0])
            if "train" in preprocess_folder:
                get_distribution_params(dist_params_dir, preprocess_folder_path)

            for i, preprocess_subfolder in enumerate(os.listdir(preprocess_folder_path)):
                if preprocess_subfolder.endswith(".pkl"):
                    continue

                preprocess_subfolder_path = os.path.join(preprocess_folder_path, preprocess_subfolder)
                files = [i for i in os.listdir(preprocess_subfolder_path) if os.path.splitext(i)[1] == ".pkl"]

                futures = [
                    executor.submit(
                        z_score_normalize_file,
                        os.path.join(preprocess_subfolder_path, file),
                        os.path.join(dist_params_dir, Y_X_DISTR_FILE),
                        os.path.join(dist_params_dir, Y_Y_DISTR_FILE),
                    )
                    for file in files
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")

        print("Done")

    except KeyboardInterrupt:
        print("Interrupted: terminating workers")
        for f in futures:
            f.cancel()
        sys.exit(1)

    finally:
        executor.shutdown()
