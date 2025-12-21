import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_path_config import DATA_PATH, DATA_VIZUALIZATION_PATH


# data in pickle:
# 0: x: [vehicle, 7]: [[x_0, y_0, speed_0, yaw_0, intent, intent, intent]]
# 1: y: [vehicle, PRED_LEN * 6]: [[x_1, y_1, v_1, yaw_1, acc_1, delta_1, x_2, y_2...], ...]
# 2: edge_indexs
# 3: timestamp
def get_data_frame_pkl(pkl_dir="csv/train_pre"):
    pkl_dir_path = os.path.join(DATA_PATH, pkl_dir)
    file_names = os.listdir(pkl_dir_path)
    print("len of generator:", len(file_names))

    for file_name in file_names:
        if not file_name.endswith(".pkl"):
            continue

        file_path = os.path.join(pkl_dir_path, file_name)
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        x = data[0].numpy()
        y = data[1].numpy()
        yield (
            x[:, 0].flatten(),
            x[:, 1].flatten(),
            x[:, 2].flatten(),
            x[:, 3].flatten(),
            y[:, 0::6].flatten(),
            y[:, 1::6].flatten(),
            y[:, 2::6].flatten(),
            y[:, 3::6].flatten(),
        )


def get_data_pkl():
    data_dict = {
        "x_xs": [],
        "x_ys": [],
        "x_speeds": [],
        "x_yaws": [],
        "y_xs": [],
        "y_ys": [],
        "y_speeds": [],
        "y_yaws": [],
    }
    for data in get_data_frame_pkl():
        x_x, x_y, x_speed, x_yaw, y_x, y_y, y_speed, y_yaw = data
        data_dict["x_xs"].append(x_x)
        data_dict["x_ys"].append(x_y)
        data_dict["x_speeds"].append(x_speed)
        data_dict["x_yaws"].append(x_yaw)
        data_dict["y_xs"].append(y_x)
        data_dict["y_ys"].append(y_y)
        data_dict["y_speeds"].append(y_speed)
        data_dict["y_yaws"].append(y_yaw)

    for key in data_dict.keys():
        data_dict[key] = np.concatenate(data_dict[key])
    return data_dict


# {
#     "TIMESTAMP": timestamp,
#     "TRACK_ID": track_id,
#     "OBJECT_TYPE": "tgt",
#     "X": x,
#     "Y": y,
#     "yaw": yaw_angle,
#     "speed": speed,
#     "CITY_NAME": "SUMO",
# }
def get_data_frame_csv(csv_dir="csv/train"):
    csv_dir_path = os.path.join(DATA_PATH, csv_dir)
    file_names = os.listdir(csv_dir_path)
    print("len of generator:", len(file_names))

    for file_name in file_names:
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(csv_dir_path, file_name)
        df = pd.read_csv(file_path)
        yield df["X"].to_numpy().flatten(), df["Y"].to_numpy().flatten(), df["speed"].to_numpy().flatten(), df["yaw"].to_numpy().flatten()


def get_data_csv():
    data_dict = {
        "x_xs": [],
        "x_ys": [],
        "x_speeds": [],
        "x_yaws": [],
    }
    for data in get_data_frame_csv():
        x_x, x_y, x_speed, x_yaw = data
        data_dict["x_xs"].append(x_x)
        data_dict["x_ys"].append(x_y)
        data_dict["x_speeds"].append(x_speed)
        data_dict["x_yaws"].append(x_yaw)

    for key in data_dict.keys():
        data_dict[key] = np.concatenate(data_dict[key])
    return data_dict


def vizualize_data(pkl=False):
    os.makedirs(DATA_VIZUALIZATION_PATH, exist_ok=True)
    if pkl:
        data_dict = get_data_pkl()
    else:
        data_dict = get_data_csv()

    for key, values in data_dict.items():
        plt.hist(values, bins=50, density=False)
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(DATA_VIZUALIZATION_PATH, f"{key}.png"))
        plt.cla()


if __name__ == "__main__":
    vizualize_data(True)
