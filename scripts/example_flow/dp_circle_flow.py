"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

cs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.76, 0.77, 0.78]
df = pd.DataFrame(columns=["c", "time", "rel_error", "drag"])
for i, c in enumerate(cs):
    start = time.perf_counter()
    prob = Problem()
    corners = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=600,
        num_poles=0,
        deg_poly=80,
        spacing="linear",
    )
    z_0 = 0.5 + 0.5j
    r = np.sqrt(c / np.pi)
    circle = lambda t: z_0 + r * np.exp(2j * np.pi * t)
    circle_deriv = lambda t: 2j * np.pi * r * np.exp(2j * np.pi * t)
    prob.add_interior_curve(
        lambda t: z_0 + r * np.exp(2j * np.pi * t),
        num_points=200,
        deg_laurent=60,
        centroid=z_0,
    )
    prob.domain.plot()
    plt.show()
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
    F = sol.force(circle, circle_deriv)
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
    drag = 1 / perm
    end = time.perf_counter()
    # fig, ax = an.plot(resolution=100, interior_patch=True, enlarge_patch=1.1)
    # plt.show()
