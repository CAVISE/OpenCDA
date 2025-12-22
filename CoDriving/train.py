from os import listdir
from os.path import join, isfile
import torch.multiprocessing as mp
import torch
import itertools
import time
import psutil
from collections import deque

# nvidia-ml-py package
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from data_path_config import EXPIREMENTS_PATH, DATA_PATH, EXPIREMENTS_MODELS_CONFIG_PATH, EXPIREMENTS_TRAIN_CONFIG_PATH
from CoDriving.train_scripts.train_one_config import train_one_config


def get_gpu_usage(gpu_ind: int):
    handled_gpu = nvmlDeviceGetHandleByIndex(gpu_ind)
    gpu_info = nvmlDeviceGetMemoryInfo(handled_gpu)
    gpu_usage_percentage = gpu_info.used / gpu_info.total
    return gpu_usage_percentage


def get_max_gpu_usage_percentage(gpu_devices):
    max_gpu_usage_percentage = 0
    for gpu_ind in gpu_devices:
        max_gpu_usage_percentage = max(max_gpu_usage_percentage, get_gpu_usage(gpu_ind))
    return max_gpu_usage_percentage


def get_ram_usage():
    mem = psutil.virtual_memory()
    return mem.used / mem.total


def get_dir_config_files(dir_path: str):
    dir_config_paths = []
    for file_name in listdir(dir_path):
        file_path = join(dir_path, file_name)

        if isfile(file_path):
            dir_config_paths.append(file_path)
    return dir_config_paths


def train_many_configs(max_needed_gpu_usage=0.7, max_mem_usage=0.7, max_processes=5, wait_time=4):
    nvmlInit()

    main_device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_devices = [i for i in range(torch.cuda.device_count())]
    gpu_num = len(gpu_devices)

    train_config_paths = get_dir_config_files(EXPIREMENTS_TRAIN_CONFIG_PATH)
    model_config_paths = get_dir_config_files(EXPIREMENTS_MODELS_CONFIG_PATH)
    config_pairs = list(itertools.product(train_config_paths, model_config_paths))

    task_queue = deque(config_pairs)
    active_processes = []

    while task_queue or len(active_processes):
        while (
            task_queue
            and get_max_gpu_usage_percentage(gpu_devices) < max_needed_gpu_usage
            and get_ram_usage() < max_mem_usage
            and len(active_processes) < max_processes
        ):
            device_str = main_device
            if not gpu_num == 0:
                ind = len(active_processes) % gpu_num
                device_str = f"{device_str}:{gpu_devices[ind]}"

            config_pair = task_queue.popleft()
            p = mp.Process(target=train_one_config, args=(config_pair[0], config_pair[1], EXPIREMENTS_PATH, DATA_PATH, device_str, False))
            p.start()
            active_processes.append(p)
            time.sleep(wait_time)

        for p in active_processes[:]:
            if not p.is_alive():
                p.join()
                p.close()
                active_processes.remove(p)
        time.sleep(0.1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.cuda.empty_cache()
    train_many_configs()
