from log_normal_circles import plot_summary_data
import pandas as pd
import numpy as np
import pickle
import os
import shutil


def concatenate(project_name, lengths):
    # concatenate the data
    summary_names = [
        "error_data",
        "nc_data",
        "permeability_data",
        "porosity_data",
        "run_time_data",
        "wss_mean_data",
        "wss_std_data",
    ]
    root = f"data/{project_name}"
    out_name = "log_normal_concatenated_data"
    os.mkdir(f"{root}/{out_name}")
    os.mkdir(f"{root}/{out_name}/summary_data")
    parameters = pickle.load(
        open(f"data/{project_name}/log_normal_4_data/parameters.pkl", "rb")
    )
    parameters["lengths"] = lengths
    pickle.dump(
        parameters,
        open(f"{root}/{out_name}/parameters.pkl", "wb"),
    )
    all_summary_dfs = []
    n_max = 1000
    seeds = range(n_max)
    empty_data = np.full((len(lengths), n_max), np.nan)
    concat_summary_dfs = {
        name: pd.DataFrame(empty_data, columns=seeds, index=lengths)
        for name in summary_names
    }

    for length in lengths:
        foldername = f"log_normal_{length}_data"
        main_folder = f"{length:.1e}"
        shutil.copytree(
            f"{root}/{foldername}/{main_folder}",
            f"{root}/{out_name}/{main_folder}",
        )
        summary_dfs = {
            name: pd.read_csv(f"{root}/{foldername}/summary_data/{name}.csv")
            for name in summary_names
        }
        all_summary_dfs.append(summary_dfs)
    for name in summary_names:
        for i, length in enumerate(lengths):
            concat_summary_dfs[name].iloc[i] = all_summary_dfs[i][name].loc[0][
                1:1001
            ]
        concat_summary_dfs[name].index.name = "Lengths"
        concat_summary_dfs[name].to_csv(
            f"{root}/{out_name}/summary_data/{name}.csv"
        )


if __name__ == "__main__":
    # concatenate("log_normal_hellion", [4, 8, 10, 16])
    project_name = "log_normal_hellion/log_normal_concatenated_data"
    plot_summary_data(project_name, error_type="std")
