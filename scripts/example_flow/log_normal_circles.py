"""Simulate flow past log normally distributed circles."""
from pylars import Problem, Solver, Analysis
from pylars.domain import generate_rv_circles
from scipy.stats import lognorm
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

    for centroid, radius in zip(centers, radii):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=200,
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
    sol = solver.solve(check=False, normalize=False, weight=False)
    print(f"Residual: {sol.max_residual}")
    # print(
    #     f"Residual: {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    # )
    return sol, solver.max_residual, solver.run_time


def analyse_case(sol, radii, length=2, p_drop=0.25, plot=False):
    """Analyse the solution."""
    an = Analysis(sol)
    curve = lambda t: length + 2j * length * t - 1j * length
    curve_deriv = lambda t: 2j * np.ones_like(t) * length
    permeability = an.get_permeability(
        curve=curve, curve_deriv=curve_deriv, delta_x=length, delta_p=p_drop
    )
    curves = []
    samples = np.round(20 * radii / np.min(radii))
    wss_data = an.get_wss_data(curves, samples)
    if plot:
        fig, ax = an.plot(
            resolution=100, interior_patch=True, enlarge_patch=1.01, epsilon=0
        )
        return fig, ax, permeability, wss_data
    else:
        return permeability, wss_data


def run(parameters):
    # load parameters
    project_name = parameters["project_name"]
    porosity = parameters["porosity"]
    L = parameters["L"]
    U = parameters["U"]
    mu = parameters["mu"]
    dist = parameters["rv"]
    rv_args = parameters["rv_args"]
    if dist == "lognorm":
        rv = lognorm.rvs
    if dist == "gamma":
        rv = lognorm.rvs
    lengths = parameters["lengths"]
    seeds = parameters["seeds"]

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
    nc_data = np.zeros((len(lengths), len(seeds)))
    porosity_data = np.zeros((len(lengths), len(seeds)))
    run_time_data = np.zeros((len(lengths), len(seeds)))
    residual_data = np.zeros((len(lengths), len(seeds)))
    permeability_data = np.zeros((len(lengths), len(seeds)))
    mean_wss_data = np.zeros((len(lengths), len(seeds)))
    p_drop = 10
    for i, length in enumerate(lengths):
        foldername = f"{length*L:.1e}"
        print(f"Starting length {foldername}")
        bound = length / 2
        os.mkdir(f"data/{project_name}/{foldername}")
        for j, seed in enumerate(seeds):
            print(" --- Starting seed", seed)
            np.random.seed(seed)
            centroids, radii = generate_rv_circles(
                porosity=porosity,
                rv=rv,
                rv_args=rv_args,
                length=length,
                min_dist=0.05,
            )
            n_circles = len(centroids)
            sol, residual, run_time = run_case(centroids, radii, bound, p_drop)
            dim_sol = sol.dimensionalize(U=U, L=L, mu=mu)
            dim_length = length * L
            dim_p_drop = p_drop * U * mu / L
            dim_radii = radii * L
            fig, ax, permeability, wss_data = analyse_case(
                dim_sol, dim_radii, dim_length, dim_p_drop, plot=True
            )
            filename = f"data/{project_name}/{foldername}/seed_{seed}"
            ax.axis("off")
            plt.savefig(filename + ".pdf", bbox_inches="tight")
            plt.close()
            np.savez(
                f"{filename}.npz",
                nc=n_circles,
                porosity=porosity,
                residual=residual,
                run_time=run_time,
                permeability=permeability,
                wss_data=wss_data,
            )
            nc_data[i, j] = n_circles
            porosity_data[i, j] = 1 - np.sum(np.pi * radii**2) / length**2
            run_time_data[i, j] = run_time
            residual_data[i, j] = residual
            permeability_data[i, j] = permeability
            mean_wss_data[i, j] = np.mean(wss_data)
            # save every sim to be safe
            data = {
                "nc": nc_data,
                "porosity": porosity_data,
                "run_time": run_time_data,
                "residual": residual_data,
                "permeability": permeability_data,
                "mean_wss": mean_wss_data,
            }
            for name, data in data.items():
                save_df(
                    data, f"data/{project_name}/summary_data/{name}_data.csv"
                )

    data = {
        "nc": nc_data,
        "porosity": porosity_data,
        "run_time": run_time_data,
        "residual": residual_data,
        "permeability": permeability_data,
        "mean_wss": mean_wss_data,
    }
    for name, data in data.items():
        save_df(data, f"data/{project_name}/summary_data/{name}_data.csv")


def plot_summary_data(project_name):
    plt.style.use("ggplot")
    parameters = pickle.load(open(f"data/{project_name}/parameters.pkl", "rb"))
    lengths = parameters["lengths"]
    L = parameters["L"]
    nc_data = pd.read_csv(
        f"data/{project_name}/summary_data/nc_data.csv"
    ).to_numpy()
    porosity_data = pd.read_csv(
        f"data/{project_name}/summary_data/porosity_data.csv"
    ).to_numpy()
    run_time_data = pd.read_csv(
        f"data/{project_name}/summary_data/run_time_data.csv"
    ).to_numpy()
    residual_data = pd.read_csv(
        f"data/{project_name}/summary_data/residual_data.csv"
    ).to_numpy()
    permeability_data_data = pd.read_csv(
        f"data/{project_name}/summary_data/permeability_data.csv"
    ).to_numpy()
    mean_wss_data_data = pd.read_csv(
        f"data/{project_name}/summary_data/mean_wss_data.csv"
    ).to_numpy()
    fig, ax = plt.subplots(3, 2)
    err_nc = np.std(nc_data, axis=1)
    ax[0, 0].errorbar(lengths * L, np.mean(nc_data, axis=1), yerr=err_nc)
    ax[0, 0].set_ylabel("Number of circles")
    ax[0, 0].set_xlabel("Length (m)")
    err_porosity = np.std(porosity_data, axis=1)
    ax[0, 1].errorbar(
        lengths * L, np.mean(porosity_data, axis=1), yerr=err_porosity
    )
    ax[0, 1].set_ylabel("Porosity")
    ax[0, 1].set_xlabel("Length (m)")
    plt.show()


if __name__ == "__main__":
    parameters = {
        "project_name": "log_normal_test",
        "porosity": 0.95,
        "L": 1,
        "U": 1,
        "mu": 1,
        "rv": "lognorm",
        "rv_args": {"s": 0.5, "scale": 0.275, "loc": 0.0},
        "lengths": [20],
        "seeds": [1],
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
