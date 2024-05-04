import os
import shutil


def proccess_directory(count):
    number = '{:06d}'.format(count)
    source_directory = "data_dumping"
    postfixes = [".pcd", ".yaml", "_camera0.png", "_camera1.png", "_camera2.png", "_camera3.png"]
    
    subdirectories = sorted([d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))])
    
    data_directory = os.path.join(source_directory, subdirectories[-2])
    subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])
    
    shutil.copy(os.path.join(data_directory, "data_protocol.yaml"), "data_dumping/sample/now")
    
    for folder in subdirectories:
        destination_folder = os.path.join("data_dumping/sample/now", folder)
        os.makedirs(destination_folder, exist_ok=True)
        for postfix in postfixes:
            source_file_path = os.path.join(data_directory, folder, f"{number}{postfix}")
            destination_file_path = os.path.join(destination_folder, f"{number}{postfix}")
            shutil.copy(source_file_path, destination_file_path)


def clear_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                if os.path.isfile(sub_item_path):
                    os.remove(sub_item_path)
                elif os.path.isdir(sub_item_path):
                    shutil.rmtree(sub_item_path)


def clear_directory_now(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)