import argparse
import os
import sys
from multiprocessing import Value, Lock
import concurrent.futures

from data_path_config import DATA_PATH
from CoDriving.data_scripts.preprocess_utils import process_file
from CoDriving.data_scripts.utils.base_utils import del_files_in_dir


def process_file_wrapper(*args):
    result = process_file(*args)
    with lock:
        completed_tasks_amount.value += 1
        print(f"Progress: {completed_tasks_amount.value}/{len(csv_files)}")

    return result


if __name__ == "__main__":
    cpu_amount = os.cpu_count()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalization flag",
    )
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
    parser.add_argument("--num_mpc_aug", type=int, help="number of MPC augmentation", default=0)
    parser.add_argument(
        "--processes",
        type=int,
        help=f"amount of processes(max: {cpu_amount})",
        default=cpu_amount,
    )
    parser.add_argument(
        "--intention_config",
        type=str,
        help="Name of file with routes and intentins. It must be in sumo/intentions/ directory",
        default="simple_separate_10m_intentions.json",
    )
    args = parser.parse_args()

    csv_folder = args.csv_folder
    csv_folder = os.path.join(DATA_PATH, csv_folder)
    preprocess_folder = args.pkl_folder
    preprocess_folder = os.path.join(DATA_PATH, preprocess_folder)
    os.makedirs(preprocess_folder, exist_ok=True)
    del_files_in_dir(preprocess_folder)

    n_mpc_aug = args.num_mpc_aug
    processes = args.processes
    intention_config = args.intention_config
    normalize = args.normalize

    completed_tasks_amount = Value("i", 0)
    lock = Lock()

    csv_files = [i for i in os.listdir(csv_folder) if os.path.splitext(i)[1] == ".csv"]
    intention_config_path = os.path.join(DATA_PATH, "sumo", "intentions", intention_config)

    print("Processing started")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=processes)
    try:
        futures = [
            executor.submit(
                # process_file,
                process_file_wrapper,
                csv_folder,
                file,
                preprocess_folder,
                intention_config_path,
                n_mpc_aug,
                normalize,
                True,  # allign initial direction of motion to +X
            )
            for file in csv_files
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")
        print("Done")
    except KeyboardInterrupt:
        print("\nInterrupted by the user. Termination of tasks has begun. It may take some time.")
        print("!!!DON'T INTERRUPT IT, JUST WAIT!!!")
        for f in futures:
            f.cancel()
        sys.exit(1)
    finally:
        executor.shutdown()
