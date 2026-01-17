import argparse
import os
import sys
from multiprocessing import Value, Lock
import concurrent.futures

from data_path_config import DATA_PATH
from CoDriving.data_scripts.preprocess_utils import preprocess_file
import shutil


def process_file_wrapper(*args):
    result = preprocess_file(*args)
    with lock:
        completed_tasks_amount.value += 1
        print(f"Progress: {completed_tasks_amount.value}/{len(csv_files)}")

    return result


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
    parser.add_argument(
        "--intention_config",
        type=str,
        help="Name of file with routes and intentins. It must be in sumo/intentions/ directory",
        default="simple_separate_10m_intentions.json",
    )
    parser.add_argument(
        "--start_position_config",
        type=str,
        help="Name of file with strart positions. It must be in sumo/intentions/ directory",
        default="start_positions.json",
    )
    args = parser.parse_args()

    csv_folder = args.csv_folder
    csv_folder_path = os.path.join(DATA_PATH, csv_folder)

    preprocess_folder = args.pkl_folder
    preprocess_folder_path = os.path.join(DATA_PATH, preprocess_folder)

    if os.path.exists(preprocess_folder_path):
        shutil.rmtree(preprocess_folder_path)
    os.makedirs(preprocess_folder_path, exist_ok=True)

    processes = args.processes
    intention_config = args.intention_config
    start_position_config = args.start_position_config

    completed_tasks_amount = Value("i", 0)
    lock = Lock()

    csv_files = [i for i in os.listdir(csv_folder_path) if os.path.splitext(i)[1] == ".csv"]
    intention_config_path = os.path.join(DATA_PATH, "sumo", "intentions", intention_config)
    start_position_config_path = os.path.join(DATA_PATH, "sumo", "intentions", start_position_config)

    print("Processing started")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=processes)
    try:
        futures = [
            executor.submit(
                process_file_wrapper,
                csv_folder_path,
                file,
                preprocess_folder_path,
                intention_config_path,
                start_position_config_path,
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
