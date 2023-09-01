"""Simulate flow past log normally distributed circles."""
from pylars import Problem, Solver, Analysis
from pylars.domain import generate_rv_circles
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os


def run_case(centers, radii, bound, p_drop):
    """Run doubly periodic flow past the circles."""
    prob = Problem()
    n_circles = len(centers)
    corners = [
        bound + bound * 1j,
        -bound + bound * 1j,
        -bound - bound * 1j,
        bound - bound * 1j,
    ]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=200 * n_circles,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )
    if np.max(radii) == np.min(radii):
        scales = np.ones_like(radii)
    else:
        scales = (radii - np.min(radii)) / (np.max(radii) - np.min(radii))
    for centroid, radius, s in zip(centers, radii, scales):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=100 + int(100 * s),
            deg_laurent=20 + int(20 * s),
            centroid=centroid,
            mirror_laurents=True,
            mirror_tol=bound / 2,
        )
    prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
    prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
    prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
    prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
    prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    interiors = [str(i) for i in range(4, 4 + n_circles)]
    for interior in interiors:
        prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
        prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)

    solver = Solver(prob, verbose=True)
    sol = solver.solve(check=False, normalize=False, weight=False)
    max_error = solver.max_error
    print(f"error: {max_error}")
    # print(
    #     f"error: {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    # )
    return sol, max_error, solver.run_time


def analyse_case(sol, centers, radii, length=2, p_drop=0.25, plot=False):
    """Analyse the solution."""
    an = Analysis(sol)
    curve = lambda t: length / 2 + 2j * length / 2 * t - 1j * length / 2
    curve_deriv = lambda t: 2j * np.ones_like(t) * length / 2
    permeability = an.get_permeability(
        curve=curve, curve_deriv=curve_deriv, delta_x=length, delta_p=p_drop
    )
    outlet_profile = sol.uv(curve(np.linspace(0, 1, 200)))
    curves = [
        lambda t: center + radius * np.exp(2j * np.pi * t)
        for center, radius in zip(centers, radii)
    ]
    samples = np.round(100 * radii / np.min(radii))
    surface_length = np.sum([2 * np.pi * r for r in radii])
    wss_data = an.get_wss_data(curves, samples)
    wss_mean = np.ma.mean(np.abs(wss_data))
    wss_std = np.std(wss_data)
    if plot:
        fig, ax = an.plot(
            resolution=100, interior_patch=True, enlarge_patch=1.01, epsilon=0
        )
        return (
            fig,
            ax,
            permeability,
            outlet_profile,
            wss_mean,
            wss_std,
            wss_data,
            surface_length,
        )
    else:
        return permeability, wss_data


def run(parameters):
    # load parameters
    project_name = parameters["project_name"]
    porosity = parameters["porosity"]
    n_max = parameters["n_max"]
    alpha = parameters["alpha"]
    eps_CLT = parameters["eps_CLT"]
    dist = parameters["rv"]
    rv_args = parameters["rv_args"]
    if dist == "lognorm":
        rv = lognorm.rvs
    if dist == "gamma":
        rv = lognorm.rvs
    lengths = parameters["lengths"]
    seeds = np.arange(n_max)
    err_tol = 1e-2

    def save_df(data, filename):
        df = pd.DataFrame(data, columns=seeds, index=lengths)
        df.index.name = "Lengths"
        df.to_csv(filename)

    try:
        os.mkdir(f"data/{project_name}")
    except FileExistsError:
        raise FileExistsError("Please delete the folder manually")
    os.mkdir(f"data/{project_name}/summary_data")
    with open(f"data/{project_name}/parameters.pkl", "wb") as f:
        pickle.dump(parameters, f)
    nc_data = np.full((len(lengths), n_max), np.nan)
    porosity_data = np.full((len(lengths), n_max), np.nan)
    run_time_data = np.full((len(lengths), n_max), np.nan)
    error_data = np.full((len(lengths), n_max), np.nan)
    permeability_data = np.full((len(lengths), n_max), np.nan)
    wss_mean_data = np.full((len(lengths), n_max), np.nan)
    wss_std_data = np.full((len(lengths), n_max), np.nan)
    p_drop = 10
    for i, length in enumerate(lengths):
        foldername = f"{length:.1e}"
        print(f"Starting length {foldername}")
        bound = length / 2
        os.mkdir(f"data/{project_name}/{foldername}")
        n = 0
        converged_steps = 0
        for n in range(n_max):
            seed = n + i * n_max
            print(" --- Starting sample", n)
            np.random.seed(seed)
            centroids, radii = generate_rv_circles(
                porosity=porosity,
                rv=rv,
                rv_args=rv_args,
                length=length,
                min_dist=0.05,
            )
            n_circles = len(centroids)
            sol, error, run_time = run_case(centroids, radii, bound, p_drop)
            (
                fig,
                ax,
                permeability,
                outlet_profile,
                wss_mean,
                wss_std,
                wss_data,
                surface_length,
            ) = analyse_case(
                sol, centroids, radii, length=length, p_drop=p_drop, plot=True
            )
            filename = f"data/{project_name}/{foldername}/seed_{seed}"
            ax.axis("off")
            plt.savefig(filename + ".pdf", bbox_inches="tight")
            plt.close()
            np.savez(
                f"{filename}.npz",
                nc=n_circles,
                porosity=porosity,
                error=error,
                run_time=run_time,
                permeability=permeability,
                outlet_profile=outlet_profile,
                wss_mean=wss_mean,
                wss_std=wss_std,
                wss_data=wss_data,
                surface_length=surface_length,
            )
            if error > err_tol:
                print(
                    f"Error {error} too large on iteration {n}. Skipping iteration."
                )
                continue
            nc_data[i, n] = n_circles
            porosity_data[i, n] = 1 - np.sum(np.pi * radii**2) / length**2
            run_time_data[i, n] = run_time
            error_data[i, n] = error
            permeability_data[i, n] = permeability
            wss_mean_data[i, n] = wss_mean
            wss_std_data[i, n] = wss_std
            # save every sim to be safe
            data = {
                "nc": nc_data,
                "porosity": porosity_data,
                "run_time": run_time_data,
                "error": error_data,
                "permeability": permeability_data,
                "wss_mean": wss_mean_data,
                "wss_std": wss_std_data,
            }
            for name, data in data.items():
                save_df(
                    data, f"data/{project_name}/summary_data/{name}_data.csv"
                )
            mean_perm = np.ma.mean(permeability_data[i, : n + 1])
            sigma_perm = np.std(permeability_data[i, : n + 1])
            n_crit_perm = (sigma_perm / (eps_CLT * mean_perm)) ** 2 * (
                norm.ppf(1 - alpha / 2)
            ) ** 2
            mean_wssm = np.ma.mean(wss_mean_data[i, : n + 1])
            sigma_wssm = np.std(wss_mean_data[i, : n + 1])
            n_crit_wssm = (sigma_wssm / (eps_CLT * mean_wssm)) ** 2 * (
                norm.ppf(1 - alpha / 2)
            ) ** 2
            n_crit = np.max([n_crit_perm, n_crit_wssm])
            if n > n_crit:
                converged_steps += 1
                print(f"Converged {converged_steps} times on iteration {n}.")
            if n < n_crit and converged_steps > 0:
                converged_steps = 0
            if converged_steps > 3:
                break

    data = {
        "nc": nc_data,
        "porosity": porosity_data,
        "run_time": run_time_data,
        "error": error_data,
        "permeability": permeability_data,
        "wss_mean": wss_mean_data,
        "wss_std": wss_std_data,
    }
    for name, data in data.items():
        save_df(data, f"data/{project_name}/summary_data/{name}_data.csv")


def plot_summary_data(project_name, error_type="Confidence"):
    plt.style.use("ggplot")
    parameters = pickle.load(open(f"data/{project_name}/parameters.pkl", "rb"))
    lengths = parameters["lengths"]
    og_error_data = pd.read_csv(
        f"data/{project_name}/summary_data/error_data.csv"
    ).to_numpy()[:, 1:]
    converged = np.where(og_error_data < 1e-2, True, False)
    error_data = np.ma.array(og_error_data, mask=~converged)
    num_samples = np.sum(converged, axis=1)
    error_kw_confidence = dict(ecolor="blue", lw=1, capsize=5, capthick=1)
    err_sf = norm.ppf(1 - 0.05 / 2) / np.sqrt(num_samples)
    label_confidence = "95 % CI"
    error_kw_std = dict(ecolor="black", lw=1, capsize=5, capthick=1)
    label_std = "Std. Dev."

    def get_data(filename):
        data = pd.read_csv(filename).to_numpy()[:, 1:]
        data = np.ma.array(data, mask=~converged)
        return data

    nc_data = get_data(f"data/{project_name}/summary_data/nc_data.csv")
    porosity_data = get_data(
        f"data/{project_name}/summary_data/porosity_data.csv"
    )
    run_time_data = get_data(
        f"data/{project_name}/summary_data/run_time_data.csv"
    )
    permeability_data = get_data(
        f"data/{project_name}/summary_data/permeability_data.csv"
    )
    wss_mean_data = get_data(
        f"data/{project_name}/summary_data/wss_mean_data.csv"
    )

    fig, ax = plt.subplots(3, 2, sharex=True)
    sig_nc = np.std(nc_data, axis=1)
    err_nc = sig_nc * err_sf
    ax[0, 0].errorbar(
        lengths,
        np.ma.mean(nc_data, axis=1),
        yerr=sig_nc,
        **error_kw_confidence,
    )
    ax[0, 0].errorbar(
        lengths,
        np.ma.mean(nc_data, axis=1),
        yerr=sig_nc,
        **error_kw_confidence,
    )
    ax[0, 0].set_ylabel("Number of circles")
    sig_porosity = np.std(porosity_data, axis=1)
    err_porosity = sig_porosity * err_sf
    ax[0, 1].errorbar(
        lengths,
        np.ma.mean(porosity_data, axis=1),
        yerr=err_porosity,
        **error_kw_confidence,
    )
    ax[0, 1].set_ylabel("Porosity")

    sig_rt = np.std(run_time_data, axis=1)
    err_rt = sig_rt * err_sf
    ax[1, 0].errorbar(
        lengths,
        np.ma.mean(run_time_data, axis=1),
        yerr=err_rt,
        **error_kw_confidence,
    )
    ax[1, 0].set_ylabel("Run Time")
    sig_error = np.std(error_data, axis=1)
    err_error = sig_error * err_sf
    ax[1, 1].set_yscale("log")
    ax[1, 1].errorbar(
        lengths,
        np.ma.mean(error_data, axis=1),
        yerr=err_error,
        **error_kw_confidence,
    )
    ax[1, 1].set_ylabel("Error")

    sig_perm = np.std(permeability_data, axis=1)
    err_perm = sig_perm * err_sf
    ax[2, 0].errorbar(
        lengths,
        np.ma.mean(permeability_data, axis=1),
        yerr=err_perm,
        **error_kw_confidence,
    )
    ax[2, 0].set_ylabel("Permeability")
    ax[2, 0].set_xlabel("Length (m)")
    sig_wss = np.std(wss_mean_data, axis=1)
    err_wss = sig_wss * err_sf
    ax[2, 1].errorbar(
        lengths,
        np.ma.mean(wss_mean_data, axis=1),
        yerr=err_wss,
        **error_kw_confidence,
    )
    ax[2, 1].set_ylabel("WSS Mean")
    ax[2, 1].set_xlabel("Length (m)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parameters = {
        "project_name": "log_normal_n_crit",
        "porosity": 0.95,
        "n_max": 1,
        "alpha": 0.05,
        "eps_CLT": 1.0,
        "rv": "lognorm",
        "rv_args": {"s": 0.5, "scale": 0.275, "loc": 0.0},
        "lengths": [5],
        "p_drop": 100,
    }
    # parameters = {
    #     "project_name": "log_normal_RVE",
    #     "porosity": 0.95,
    #     "L": 1e-6,
    #     "U": 1e-6,
    #     "mu": 1e-3,
    #     "rv": "lognorm",
    #     "rv_args": {"s": 0.5, "scale": 0.275, "loc": 0.0},
    #     "lengths": np.linspace(5, 15, 5),
    #     "seeds": range(1, 5),
    #     "p_drop": 1,
    # }
    run(parameters)
    plot_summary_data(parameters["project_name"])
