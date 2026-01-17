import os
import pandas as pd
import argparse

from data_path_config import MAIN_PATH, EXPIREMENTS_PATH, LOGS_DIR_NAME


def get_top_configs_by_metrics(top_n=5, expirements_path=EXPIREMENTS_PATH):
    rows = []
    for folder_name in os.listdir(expirements_path):
        folder_path = os.path.join(expirements_path, folder_name, LOGS_DIR_NAME)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".csv"):
                continue

            try:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, header=None, sep=',')
                last_log = df.iloc[-1, -1]
                rows.append({"folder_name": folder_name, "metric": file_name, "value": last_log})

            except Exception as e:
                print(e)

    metrics_df = pd.DataFrame(rows)
    top_n_df = (
        metrics_df.sort_values("value", ascending=True)
        .groupby("metric", sort=True)
        .head(top_n)
        .sort_values(["metric", "value"], ascending=[True, False])
    )
    top_n_df.to_csv(os.path.join(expirements_path, "top_n_metrics.csv"))


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--experements_dir_path", type=str, help="", default="experements")

    args = parser.parse_args()

    expirements_path = os.path.join(MAIN_PATH, args.experements_dir_path)
    get_top_configs_by_metrics(expirements_path=expirements_path)


if __name__ == "__main__":
    main()
