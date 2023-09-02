"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.interpolate import griddata

shift = 0.0 + 0.0j
corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
corners += shift
prob = Problem()
prob.add_exterior_polygon(
    corners,
    num_edge_points=1000,
    deg_poly=100,
    num_poles=0,
    spacing="linear",
)
centroid = shift - 0.4 - 0.4j
R = 5e-1j
prob.add_interior_curve(
    lambda t: centroid + R * np.exp(2j * np.pi * t),
    num_points=500,
    deg_laurent=120,
    centroid=centroid,
    mirror_laurents=True,
    mirror_tol=1,
)
centroid2 = shift + 0.2 + 0.5j
R = 5e-2j
prob.add_interior_curve(
    lambda t: centroid2 + R * np.exp(2j * np.pi * t),
    num_points=400,
    deg_laurent=100,
    centroid=centroid2,
    mirror_laurents=True,
    mirror_tol=2,
)
prob.domain.plot()
plt.show()

# top and bottom periodicity
p_drop = 2.0 / 0.15
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

prob.add_boundary_condition("1", "psi[1]-psi[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "u[5]", 0)
prob.add_boundary_condition("5", "v[5]", 0)

solver = Solver(prob, verbose=True)
sol = solver.solve(check=False, weight=False, normalize=False)
print(f"Error: {solver.max_error}")
an = Analysis(sol)
fig, ax = an.plot_errors(solver.errors)
plt.show()
fig, ax = an.plot(resolution=200, interior_patch=True)
errors = solver.errors
ax.axis("off")
plt.savefig("media/multiscalar.pdf", bbox_inches="tight")
plt.show()
