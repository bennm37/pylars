"""Flow a domain with a circular interior curve."""

from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def generate_data(cs, plot=False):
    small = {
        "num_points": 200,
        "deg_laurent": 50,
    }
    big = {
        "num_points": 1000,
        "deg_laurent": 50,
        "mirror_laurents": True,
        "mirror_tol": 1.5,
    }
    params = [small if c <= 0.7 else big for c in cs]
    n_cs = len(cs)
    filename = "data/dp_circle_drag.csv"
    df = pd.DataFrame(columns=["c", "time", "rel_error", "digits", "drag"])
    for i, c in enumerate(cs):
        print(f"Starting iteration {i}/{n_cs} ...")
        start = time.perf_counter()
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
        circle = lambda t: z_0 + r * np.exp(2j * np.pi * t)
        circle_deriv = lambda t: 2j * np.pi * r * np.exp(2j * np.pi * t)
        prob.add_interior_curve(
            lambda t: z_0 + r * np.exp(2j * np.pi * t), centroid=z_0, **params[i]
        )
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
        # an = Analysis(sol)
        # an.plot()
        # plt.show()
        right = lambda t: 1 + 1j * t
        right_deriv = lambda t: 1j
        delta_x = np.abs(corners[0].real - corners[1].real)
        an = Analysis(sol)
        perm = an.get_permeability(
            right,
            right_deriv,
            delta_x,
            delta_p,
        )
        max_error = solver.get_error()[0]
        psi_top = sol.psi(1 + 1j)[0][0]
        psi_bottom = sol.psi(1)[0][0]
        perm_psi = (psi_top - psi_bottom) / delta_p
        eps = 2 * max_error / delta_p
        if np.abs(perm_psi) < 2 * eps:
            rel_error = np.inf
        else:
            rel_error = np.abs(eps / (perm_psi))
        digits = int(-np.log10(rel_error))
        print(f"Accurate to within {int(digits)} digits.")
        drag = 1 / perm_psi
        print(f"{c = }, {drag = }")
        end = time.perf_counter()
        fig, ax = an.plot()
        df.loc[i] = [c, end - start, rel_error, digits, drag]
        if plot:
            fig, ax = an.plot(resolution=100, interior_patch=True, enlarge_patch=1.0)
            plt.show()
        df.to_csv(filename)
    return filename


def colored_float(num, digits):
    snum = str(num)
    black = "".join([d for d in snum])
    red = "".join([d for d in snum])
    return black + "\textcolor\{red\}" + f"{{ {red} }}"


def format_drag(num, digits):
    digits = int(digits)
    formatted_num = eval(f'f"{{ num:.{digits}g}}"')  # meta f string hack
    return formatted_num


if __name__ == "__main__":
    cs = [0.6, 0.7, 0.75, 0.76, 0.77, 0.78]
    # cs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.76, 0.77, 0.78]
    generate_data(cs)
    barnett_data = pd.read_csv("data/barnett_circle_drag.csv")
    pylars_data = pd.read_csv("data/dp_circle_drag.csv")
    print(barnett_data)
    # aggregate data
    aggregated = pylars_data.copy()
    aggregated = aggregated.rename(columns={"drag": "D_pylars"})
    aggregated["D_pylars"] = [
        format_drag(drag, digits)
        for drag, digits in zip(aggregated["D_pylars"], aggregated["digits"])
    ]
    aggregated["D_barnett"] = barnett_data["D_barn"]
    aggregated["D_GK"] = barnett_data["D_GK"]
    print(aggregated)
    # save as a latex table
