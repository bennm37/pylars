"""Flow a domain with a circular interior curve."""

from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")


def generate_circle_data(cs, filename=None):
    n_cs = len(cs)
    if filename is None:
        filename = "data/dp_circle_permeability.csv"
    df = pd.DataFrame(columns=["c", "rel_error", "digits", "permeability"])
    params = {
        "num_points": 500,
        "deg_laurent": 50,
        "mirror_laurents": False,
        "mirror_tol": 1.5,
    }
    for i, c in enumerate(cs):
        print(f"Starting iteration {i+1}/{n_cs} ...")
        prob = Problem()
        corners = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
        prob.add_exterior_polygon(
            corners,
            num_edge_points=1000,
            num_poles=0,
            deg_poly=50,
            spacing="linear",
        )
        z_0 = 0.5 + 0.5j
        r = np.sqrt(c / np.pi)
        prob.add_interior_curve(
            lambda t: z_0 + r * np.exp(2j * np.pi * t), centroid=z_0, **params
        )
        rel_error, digits, perm = get_results(prob)
        df.loc[i] = [c, rel_error, digits, perm]
        df.to_csv(filename)
    return filename


def generate_square_data(cs, filename=None):
    n_cs = len(cs)
    if filename is None:
        filename = "data/dp_square_permeability.csv"
    df = pd.DataFrame(columns=["c", "rel_error", "digits", "permeability"])
    params = {"num_points": 500, "num_poles": 24, "spacing": "clustered"}
    for i, c in enumerate(cs):
        print(f"Starting iteration {i+1}/{n_cs} ...")
        prob = Problem()
        corners = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
        prob.add_exterior_polygon(
            corners,
            num_edge_points=1000,
            num_poles=0,
            deg_poly=50,
            spacing="linear",
        )
        s = np.sqrt(c)
        z0 = 0.5 + 0.5j
        corners = [
            z0 - s / 2 - s / 2 * 1j,
            z0 + s / 2 - s / 2 * 1j,
            z0 + s / 2 + s / 2 * 1j,
            z0 - s / 2 + s / 2 * 1j,
        ]
        prob.add_interior_polygon(corners, **params)
        rel_error, digits, perm = get_results(prob)
        df.loc[i] = [c, rel_error, digits, perm]
        df.to_csv(filename)
    return filename


def get_results(prob):
    delta_p = 1
    prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
    prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
    prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
    prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)
    prob.add_boundary_condition("1", "u[3]-u[1][::-1]", 0)
    prob.add_boundary_condition("1", "v[3]-v[1][::-1]", 0)
    prob.add_boundary_condition("3", "p[3]+e11[3]-p[1][::-1]-e11[1][::-1]", delta_p)
    prob.add_boundary_condition("3", "e12[3]-e12[1][::-1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)

    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False)
    max_error = solver.max_error
    psi_top = sol.psi(1 + 1j)[0][0]
    psi_bottom = sol.psi(1)[0][0]
    perm_psi = (psi_top - psi_bottom) / delta_p
    abs_error = 2 * max_error / delta_p
    rel_error = abs_error / (abs(perm_psi) - abs_error)
    digits = int(-np.log10(rel_error))
    print(f"Accurate to within {int(digits)} digits.")
    return rel_error, digits, perm_psi


def plot_permeability(data):
    data = pd.read_csv(filename)
    fig, ax = plt.subplots()
    ax.plot(data["c"], data["permeability"], "o")
    ax.set(yscale="log")
    ax.set_xlabel("c")
    ax.set_ylabel("Permeability")
    plt.show()


if __name__ == "__main__":
    cs = np.linspace(0.01, 0.7, 20)
    filename = "data/dp_permeability_data/dp_circle_permeability.csv"
    # generate_circle_data(
    #     cs, "data/dp_permeability_data/dp_circle_permeability.csv"
    # )
    data = pd.read_csv(filename)
    plot_permeability(data)
