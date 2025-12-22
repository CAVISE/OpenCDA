import os


def del_files_in_dir(dir_path: str):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)
