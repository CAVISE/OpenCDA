import os
import pandas as pd

from .data_path_config import path_config


def get_top_configs_by_metrics(top_n: int = 5, expirements_path: str | None = None) -> None:
    """
    get top n configurations by metrics and save to csv

    :param top_n: number of top configurations to select per metric
    :param expirements_path: path to experiments directory
    """
    if expirements_path is None:
        expirements_path = path_config.paths.expirements_path

    rows = []
    for folder_name in os.listdir(expirements_path):
        folder_path = os.path.join(expirements_path, folder_name, path_config.dir_names.logs_dir_name)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".csv"):
                continue

            try:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, header=None, sep=",")
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


def main() -> None:
    """
    main function to get top configurations by metrics
    """
    get_top_configs_by_metrics()


if __name__ == "__main__":
    main()
