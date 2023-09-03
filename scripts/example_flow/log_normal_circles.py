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
        num_edge_points=120 * n_circles,
        num_poles=0,
        deg_poly=50,
        spacing="linear",
    )

    for centroid, radius in zip(centers, radii):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=120,
            deg_laurent=40,
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
    try:
        sol = solver.solve(check=False, normalize=False, weight=False)
    except ValueError:
        print(f"Solve failed.")
        return None, None, None
    max_error = solver.max_error
    print(f"error: {max_error}")
    # print(
    #     f"error: {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    # )
    return sol, max_error, solver.run_time


def analyse_case(sol, centers, radii, length=2, p_drop=0.25, plot=True):
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
    derivs = [
        lambda t: 2j * np.pi * radius * np.exp(2j * np.pi * t)
        for center, radius in zip(centers, radii)
    ]
    samples = np.round(100 * radii / np.min(radii))
    surface_length = np.sum([2 * np.pi * r for r in radii])
    wss_data = an.get_wss_data(curves, derivs, samples)
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
        return (
            None,
            None,
            permeability,
            outlet_profile,
            wss_mean,
            wss_std,
            wss_data,
            surface_length,
        )


def run(parameters):
    # load parameters
    project_name = parameters["project_name"]
    porosity = parameters["porosity"]
    n_max = parameters["n_max"]
    alpha = parameters["alpha"]
    eps_CLT = parameters["eps_CLT"]
    rv = parameters["rv"]
    rv_args = parameters["rv_args"]
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
        save_parameters = parameters.copy()
        save_parameters.pop("rv")
        pickle.dump(save_parameters, f)
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
            # np.random.seed(seed)
            centroids, radii = generate_rv_circles(
                porosity=porosity,
                rv=rv,
                rv_args=rv_args,
                length=length,
                min_dist=0.1,
            )
            n_circles = len(centroids)
            sol, error, run_time = run_case(centroids, radii, bound, p_drop)
            if sol is None:
                print("Solve failed. Skipping iteration {n}.")
                continue
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
            if fig is not None:
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
            p_data = permeability_data[i, : n + 1].copy()
            p_data = np.delete(p_data, np.where(np.isnan(p_data)))
            mean_perm = np.ma.mean(p_data)
            sigma_perm = np.std(p_data)
            n_crit_perm = (sigma_perm / (eps_CLT * mean_perm)) ** 2 * (
                norm.ppf(1 - alpha / 2)
            ) ** 2
            wssm_data = wss_mean_data[i, : n + 1].copy()
            wssm_data = np.delete(wssm_data, np.where(np.isnan(wssm_data)))
            mean_wssm = np.ma.mean(wssm_data)
            sigma_wssm = np.std(wssm_data)
            n_crit_wssm = (sigma_wssm / (eps_CLT * mean_wssm)) ** 2 * (
                norm.ppf(1 - alpha / 2)
            ) ** 2
            n_crit = np.max([n_crit_perm, n_crit_wssm])
            print(f"n_crit estimate is {n_crit}")
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


def plot_summary_data(project_name, scale=None):
    plt.style.use("ggplot")
    parameters = pickle.load(open(f"data/{project_name}/parameters.pkl", "rb"))
    lengths = np.array(parameters["lengths"])
    og_error_data = pd.read_csv(
        f"data/{project_name}/summary_data/error_data.csv"
    ).to_numpy()[:, 1:]
    converged = np.where(og_error_data < 1e-2, True, False)
    error_data = np.ma.array(og_error_data, mask=~converged)
    num_samples = np.sum(converged, axis=1)
    error_kw_confidence = dict(ecolor="blue", lw=1, capsize=5, capthick=1)
    err_sf = norm.ppf(1 - 0.05 / 2) / np.sqrt(num_samples)
    label_confidence = "95 \% CI"
    error_kw_std = dict(ecolor="red", lw=1, capsize=5, capthick=1)
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
    wss_std_data = get_data(
        f"data/{project_name}/summary_data/wss_std_data.csv"
    )
    if scale is not None:
        L, U, mu, grad_p = scale
        # the simulations are run with a non-dimensional pressure drop of 2.
        # To impose a pressure gradient of grad_p Pa, we need to scale
        # by grad_p * non_dim_l * U * mu/ (2 * L)
        wss_mean_data = (
            grad_p * lengths[:, np.newaxis] * wss_mean_data * mu * U / (10 * L)
        )
        wss_std_data = (
            grad_p * lengths[:, np.newaxis] * wss_std_data * mu * U / (10 * L)
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
        ax.errorbar(
            lengths,
            mean,
            yerr=sig,
            color="black",
            **error_kw_std,
            label=label_std,
        )
        ax.set_ylabel(name)

    fig, ax = plt.subplots(3, 2, sharex=True)
    plot("Number of Circles", nc_data, ax[0, 0])
    plot("Porosity", porosity_data, ax[0, 1])
    plot("Run Time", run_time_data, ax[1, 0])
    plot("Rel Error", error_data, ax[1, 1])
    ax[1, 1].set_yscale("log")
    plot("Permeability ($\mathrm{m}^2$)", permeability_data, ax[2, 0])
    plot("WSS Mean", wss_mean_data, ax[2, 1])

    ax[2, 1].set_ylabel("WSS Mean (Pa)")
    ax[2, 1].set_xlabel("Length (m)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    mean, var = lognorm.stats(
        **{"s": 0.5, "scale": 0.275, "loc": 0.0}, moments="mv"
    )
    rv = lambda mu: mu
    rv_args = {"mu": mean}
    parameters = {
        "project_name": "uniform_circles",
        "porosity": 0.95,
        "n_max": 300,
        "alpha": 0.05,
        "eps_CLT": 0.05,
        "rv": rv,
        "rv_args": rv_args,
        "lengths": [4, 8, 12, 16],
        "p_drop": 100,
    }
    # parameters = {
    #     "project_name": "random_test",
    #     "porosity": 0.95,
    #     "n_max": 300,
    #     "alpha": 0.05,
    #     "eps_CLT": 0.1,
    #     "rv": "lognorm",
    #     "rv_args": {"s": 0.5, "scale": 0.275, "loc": 0.0},
    #     "lengths": [8],
    #     "p_drop": 100,
    # }
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
    plot_summary_data(parameters["project_name"], scale=(1e-6, 1e-6, 1e-3, 1))
    plt.show()
