import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
# from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_map import preprocess_map
import scipy.stats as stats


def check_angle(ax, vechile_id: str, whole_df, idx=20, c_idx=20):
    df = whole_df.loc[whole_df["TRACK_ID"] == vechile_id]
    df = df.reset_index()

    coords = df[["X", "Y"]].to_numpy()
    coords[:, 1] = -coords[:, 1]
    yaw_true = np.array(df["yaw"][idx])

    length = 5
    dcoords = coords - coords[idx]
    norms = (dcoords[:, 0] ** 2 + dcoords[:, 1] ** 2) ** 0.5 - length
    start_idx = np.argmin(np.abs(norms[:idx]))

    end_id = start_idx + 1
    if norms[start_idx] < 0:
        end_id = start_idx - 1

    kx, ky = coords[end_id] - coords[start_idx]
    bx, by = coords[start_idx] - coords[idx]

    a = kx**2 + ky**2
    b = 2 * (kx * bx + ky * by)
    c = bx**2 + by**2 - length**2

    p = b**2 / (4 * a**2) - c / a
    m = b / (2 * a)
    t1 = p**0.5 - m
    t2 = -(p**0.5) - m

    if 0 < t1 < 1:
        t = t1
    else:
        t = t2

    vector = (coords[end_id] - coords[start_idx]) * t + coords[start_idx]
    dot_vec = coords[idx] - vector
    yaw = np.arctan2(dot_vec[1], dot_vec[0]) * 180 / np.pi
    print(t, vector, yaw_true, yaw_true - yaw - 90)

    ax.plot([coords[end_id][0], coords[start_idx][0]], [coords[end_id][1], coords[start_idx][1]], linewidth=1)
    ax.scatter([coords[end_id][0], coords[start_idx][0]], [coords[end_id][1], coords[start_idx][1]], s=1)
    print(idx)
    if idx == c_idx or idx + 1 == c_idx:
        print("<<<<")
        circle = plt.Circle(coords[idx], length, fill=False)
        ax.scatter(coords[idx][0], coords[idx][1], c="black", s=20)
        ax.add_patch(circle)

    df.loc[idx, "yaw_delta"] = yaw_true - yaw - 90
    df.loc[idx, "yaw_next_delta"] = yaw_true - df.loc[idx + 1, "yaw"]
    df.loc[idx, "yaw_prev_delta"] = yaw_true - df.loc[idx - 1, "yaw"]

    ax.scatter(vector[0], vector[1], c="r", s=5)
    ax.scatter(
        coords[idx][0] - length * np.cos((yaw_true - 90) * np.pi / 180), coords[idx][1] - length * np.sin((yaw_true - 90) * np.pi / 180), c="b", s=5
    )

    # После всех расчетов, когда df готов
    whole_df.loc[whole_df["TRACK_ID"] == vechile_id, ["yaw_delta", "yaw_next_delta", "yaw_prev_delta"]] = df[
        ["yaw_delta", "yaw_next_delta", "yaw_prev_delta"]
    ].values

    return whole_df  # или return df, если нужно вернуть только подмножество
    # return df


def draw_diff(vechile_id: str, whole_df, offset=10):
    ax = plt.subplot()
    ax.axis("equal")

    df = whole_df.loc[whole_df["TRACK_ID"] == vechile_id]
    df = df.reset_index()
    yaw_true = np.array(df["yaw"])

    coords = df[["X", "Y"]].to_numpy()
    coords[:, 1] = -coords[:, 1]

    length = 5
    dcoords = coords[offset:, np.newaxis, :] - coords[np.newaxis, :, :]

    norms = (dcoords[:, :, 0] ** 2 + dcoords[:, :, 1] ** 2) ** 0.5 - length
    norms_mask = np.tri(norms.shape[0], norms.shape[1], offset - 1, dtype=bool)
    masked_norms = np.ma.array(np.abs(norms), mask=~norms_mask)
    start_idxs = masked_norms.argmin(axis=-1, fill_value=np.inf)

    end_idxs = start_idxs + 1
    zero_mask = norms[np.arange(start_idxs.shape[0]), start_idxs] < 0
    end_idxs[zero_mask] = start_idxs[zero_mask] - 1
    end_idxs_mask_idxs = np.where(end_idxs < 0)[0]

    k = coords[end_idxs, :] - coords[start_idxs, :]
    kx, ky = k[:, 0:1], k[:, 1:2]
    b = coords[start_idxs, :] - coords[offset:, :]
    bx, by = b[:, 0:1], b[:, 1:2]

    a1 = kx**2 + ky**2
    b1 = 2 * (kx * bx + ky * by)
    c1 = bx**2 + by**2 - length**2

    p = b1**2 / (4 * a1**2) - c1 / a1
    m = b1 / (2 * a1)

    t1 = p**0.5 - m
    t2 = -(p**0.5) - m

    t1_mask = (0 < t1) & (t1 < 1)
    t = t1.copy()
    t[~t1_mask] = t2[~t1_mask]

    vector = (coords[end_idxs] - coords[start_idxs]) * t + coords[start_idxs]
    vector[end_idxs_mask_idxs, 0] = coords[end_idxs_mask_idxs + offset, 0] - length * np.cos(
        (yaw_true[end_idxs_mask_idxs + offset] - 90) * np.pi / 180
    )
    vector[end_idxs_mask_idxs, 1] = coords[end_idxs_mask_idxs + offset, 1] - length * np.sin(
        (yaw_true[end_idxs_mask_idxs + offset] - 90) * np.pi / 180
    )
    # dot_vec = coords[offset:, :] - vector
    # yaw = np.arctan2(dot_vec[:, 1], dot_vec[:, 0]) * 180 / np.pi

    ax.scatter(
        coords[offset:, 0] - length * np.cos((yaw_true[offset:] - 90) * np.pi / 180),
        coords[offset:, 1] - length * np.sin((yaw_true[offset:] - 90) * np.pi / 180),
        s=0.1,
        c="b",
    )
    ax.scatter(
        vector[:, 0],
        vector[:, 1],
        s=0.1,
        c="r",
    )
    plt.savefig(".kirill_test/test_drawing.png")
    plt.cla()


def draw_sumo(vechile_id: str, whole_df):
    df = whole_df.loc[whole_df["TRACK_ID"] == vechile_id]
    df = df.reset_index()

    coords = df[["X", "Y"]].to_numpy()
    coords[:, 1] = -coords[:, 1]
    yaw_true = np.array(df["yaw"])

    length = 5
    ax = plt.subplot()
    ax.axis("equal")

    df = whole_df.loc[whole_df["TRACK_ID"] == vechile_id]
    df = df.reset_index()

    coords = df[["X", "Y"]].to_numpy()
    coords[:, 1] = -coords[:, 1]
    yaw_true = np.array(df["yaw"])

    length = 5
    ax.plot(coords[:, 0], coords[:, 1], c="r", linewidth=1)
    ax.plot(
        coords[:, 0] - length * np.cos((yaw_true - 90) * np.pi / 180),
        coords[:, 1] - length * np.sin((yaw_true - 90) * np.pi / 180),
        c="b",
        linewidth=1,
    )
    plt.savefig(".kirill_test/test_drawing.png")
    plt.cla()

    plt.hist(coords[:, 0] - coords[:, 0] + length * np.cos((yaw_true - 90) * np.pi / 180), bins=30)
    plt.savefig(".kirill_test/test_drawing_histX.png")
    plt.cla()

    plt.hist(coords[:, 1] - coords[:, 1] + length * np.sin((yaw_true - 90) * np.pi / 180), bins=30)
    plt.savefig(".kirill_test/test_drawing_histY.png")
    plt.cla()


# def check_angle_lane(lane_points, vechile_id: str, csv_path="data/csv/train/00133-00138.csv", idx=20, c_idx=20):

#     df = whole_df.loc[whole_df["TRACK_ID"] == vechile_id]
#     df = df.reset_index()

#     coords = df[["X", "Y"]].to_numpy()
#     coords[:, 1] = -coords[:, 1]
#     lane_points[:, 1] = -lane_points[:, 1]
#     yaw_true = np.array(df["yaw"][idx])

#     length = 5
#     dcoords = lane_points - coords[idx]
#     norms = (dcoords[:, 0] ** 2 + dcoords[:, 1] ** 2) ** 0.5 - length
#     start_idx = np.argmin(np.abs(norms[:idx]))

#     end_id = start_idx + 1
#     if norms[start_idx] < 0:
#         end_id = start_idx - 1

#     kx, ky = lane_points[end_id] - lane_points[start_idx]
#     bx, by = lane_points[start_idx] - coords[idx]

#     a = kx**2 + ky**2
#     b = 2 * (kx * bx + ky * by)
#     c = bx**2 + by**2 - length**2

#     p = b**2 / (4 * a**2) - c / a
#     m = b / (2 * a)
#     t1 = p**0.5 - m
#     t2 = -(p**0.5) - m

#     if 0 < t1 < 1:
#         t = t1
#     else:
#         t = t2

#     # fig, ax = plt.subplots()
#     vector = (lane_points[end_id] - lane_points[start_idx]) * t + lane_points[start_idx]
#     dot_vec = coords[idx] - vector
#     yaw = np.arctan2(dot_vec[1], dot_vec[0]) * 180 / np.pi
#     df['yaw_delta'][idx] = yaw_true - yaw - 90
#     df['yaw_next_delta'][idx] = yaw_true - df["yaw"][idx + 1]
#     df['yaw_prev_delta'][idx] = yaw_true - df["yaw"][idx - 1]

#     print(t, vector, yaw_true - yaw - 90)

#     # ax.plot([coords[end_id][0], coords[start_idx][0]], [coords[end_id][1], coords[start_idx][1]])
#     # ax.scatter([coords[end_id][0], coords[start_idx][0]], [coords[end_id][1], coords[start_idx][1]], s=2)
#     # circle = plt.Circle(coords[idx], 5, fill=False)

#     # ax.add_patch(circle)
#     # ax.scatter(vector[0], vector[1], c="r", s=5)
#     # ax.set_aspect("equal")
#     # plt.grid()
#     # plt.savefig("test_angle.png")


def draw_diff_dots():
    ax = plt.subplot()
    ax.axis("equal")

    n1 = 10
    n2 = 35
    result_df = None

    dir_path = "AIM/models/mtp/learning/data/csv/train/simple_separate_10m"
    files = ["00027-00032.csv", "00271-00276.csv", "00427-00432.csv", "00430-00435.csv"]
    car_idxs = ["E6_-E7_6", "E7_-E4_75", "E4_-E5_127", "E4_-E5_127"]
    myzip = zip(files, car_idxs)

    for file, car_idx in myzip:
        csv_path = os.path.join(dir_path, file)
        print(">>>>>>>>", csv_path)
        whole_df = pd.read_csv(csv_path, sep=",")
        whole_df["yaw_delta"] = 0
        whole_df["yaw_next_delta"] = 0
        whole_df["yaw_prev_delta"] = 0

        for k in range(n1, n2):
            whole_df = check_angle(ax, car_idx, whole_df, k, 30)

        if result_df is None:
            result_df = whole_df.copy()
        else:
            result_df = pd.concat([result_df, result_df])

        plt.savefig(f".kirill_test/test_{car_idx}.png")
        plt.cla()

    t_critical = stats.t.ppf(q=0.95, df=result_df.shape[0])
    yaw_delta_mean = result_df["yaw_delta"].mean()
    yaw_delta_std = result_df["yaw_delta"].std()
    mask = (np.abs(result_df["yaw_delta"] - yaw_delta_mean) / yaw_delta_std) > t_critical
    result_df = result_df[~mask]

    pd.plotting.scatter_matrix(result_df, figsize=(10, 10), hist_kwds={"bins": 40})
    plt.savefig(".kirill_test/test_pairplot.png")


if __name__ == "__main__":
    dir_path = "AIM/models/mtp/learning/data/csv/train/simple_separate_10m"
    csv_path = os.path.join(dir_path, "00427-00432.csv")
    whole_df = pd.read_csv(csv_path, sep=",")

    # draw_sumo("E6_-E7_6", whole_df)
    draw_diff("E4_-E5_127", whole_df)
