from log_normal_circles import plot_summary_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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


def plot_convergence_p_wss(project_name, scale=None):
    plt.style.use("ggplot")
    parameters = pickle.load(open(f"data/{project_name}/parameters.pkl", "rb"))
    lengths = np.array(parameters["lengths"])
    og_error_data = pd.read_csv(
        f"data/{project_name}/summary_data/error_data.csv"
    ).to_numpy()[:, 1:]
    converged = np.where(og_error_data < 1e-2, True, False)
    num_samples = np.sum(converged, axis=1)
    error_kw_confidence = dict(ecolor="blue", lw=1, capsize=5, capthick=1)
    err_sf = norm.ppf(1 - 0.05 / 2) / np.sqrt(num_samples)
    label_confidence = "95 \% CI"

    def get_data(filename):
        data = pd.read_csv(filename).to_numpy()[:, 1:]
        data = np.ma.array(data, mask=np.isnan(data))
        return data

    permeability_data = get_data(
        f"data/{project_name}/summary_data/permeability_data.csv"
    )
    wss_mean_data = get_data(
        f"data/{project_name}/summary_data/wss_mean_data.csv"
    )
    if scale is not None:
        L, U, mu, grad_p = scale
        # the simulations are run with a non-dimensional pressure drop of 2.
        # To impose a pressure gradient of grad_p Pa, we need to scale
        # by grad_p * non_dim_l * U * mu/ (2 * L)
        wss_mean_data = (
            grad_p * lengths[:, np.newaxis] * wss_mean_data * mu * U / (10 * L)
        )
        lengths = lengths * L
        permeability_data = permeability_data * L**2

    def plot(name, data, ax):
        mean = np.ma.mean(data, axis=1)
        sig = np.std(data, axis=1)
        err = sig * err_sf
        ax.errorbar(
            lengths,
            mean,
            yerr=err,
            color="black",
            **error_kw_confidence,
            label=label_confidence,
        )
        # ax.errorbar(
        #     lengths,
        #     mean,
        #     yerr=sig,
        #     color="black",
        #     **error_kw_std,
        #     label=label_std,
        # )
        ax.set_ylabel(name)

    fig, ax = plt.subplots(1, 2)
    plot("Permeability ($\mathrm{m}^2$)", permeability_data, ax[0])

    plot("WSS Mean (Pa)", wss_mean_data, ax[1])
    # ax[1].set_ylabel("WSS Mean (Pa)")
    ax[0].set_xlabel("Length (m)")
    ax[1].set_xlabel("Length (m)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    return fig, ax


def plot_wss_distribution(project_name, unfinished=0, scale=None):
    parameters = pickle.load(open(f"data/{project_name}/parameters.pkl", "rb"))
    if scale is not None:
        L, U, mu, grad_p = scale
    else:
        L, U, mu, grad_p = 1, 1, 1, 1
    lengths = np.array(parameters["lengths"])[:-unfinished]
    names = [f"{length:.1e}" for length in lengths]
    wss_data = {name: [] for name in names}
    hist_data = {name: [] for name in names}
    bins = np.linspace(0, 30, 30)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    for length, name in zip(lengths, names):
        # get all the file names in f"data/{project_name}/{name}/"
        # that start with "seed_"
        filenames = os.listdir(f"data/{project_name}/{name}/")
        filenames = [
            filename
            for filename in filenames
            if filename.startswith("seed_") and filename.endswith(".npz")
        ]
        # sort the filenames
        filenames = sorted(
            filenames, key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        for filename in filenames:
            data = np.load(f"data/{project_name}/{name}/{filename}")
            wss_data_i = data["wss_data"]
            if np.any(np.isnan(wss_data_i)):
                continue
            # the simulations are run with a non-dimensional pressure drop of 10.
            # To impose a pressure gradient of grad_p Pa, we need to scale
            # by grad_p * non_dim_l * U * mu/ (10 * L)
            wss_data_i = grad_p * length * wss_data_i * mu * U / (10 * L)
            hist_data_i, _ = np.histogram(wss_data_i, bins=bins, density=True)
            wss_data[name].append(wss_data_i)
            hist_data[name].append(hist_data_i)
    N = len(lengths)
    fig, ax = plt.subplots(1, N, sharey=True, figsize=(6, 2.5))
    ax[0].set_ylabel("Probability Density")
    for i, name in enumerate(names):
        mean = np.mean(hist_data[name], axis=0)
        err = (
            1.96
            * np.std(hist_data[name], axis=0)
            / np.sqrt(len(hist_data[name]))
        )
        ax[i].plot(
            bin_centers,
            mean,
        )
        ax[i].fill_between(
            bin_centers,
            mean - err,
            mean + err,
            alpha=0.5,
        )
        ax[i].set_title(f"{lengths[i]*L:.1e}")
        if i == N // 2:
            ax[i].set_xlabel("Wall Shear Stress (Pa)")
    ax[-1].legend(labels=["Mean", "95\% CI"], loc="upper center")
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # concatenate("log_normal_hellion", [4, 8, 10, 16])
    plt.style.use("ggplot")
    # project_name = "log_norm_linear_hellion_partial"
    project_name = "uniform_circles"
    # fig, ax = plot_summary_data(project_name, scale=(1e-6, 1e-6, 1e-3, 2500))
    # plt.show()
    fig, ax = plot_convergence_p_wss(
        project_name, scale=(1e-6, 1e-6, 1e-3, 2500)
    )
    fig.set_size_inches(6, 2)
    plt.tight_layout()
    plt.savefig(f"media/{project_name}_convergence.pdf", bbox_inches="tight")
    plt.show()

    # plt.savefig("media/{project_name}_convergence.pdf")
    fig, ax = plot_wss_distribution(
        project_name, unfinished=1, scale=(1e-6, 1e-6, 1e-3, 2500)
    )
    plt.savefig(
        f"media/{project_name}_wss_distribution.pdf", bbox_inches="tight"
    )
    plt.show()
