from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

c = 0.76
params = {
    "num_points": 500,
    "deg_laurent": 50,
    "mirror_laurents": True,
    "mirror_tol": 1.5,
}
prob = Problem()
corners = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=500,
    num_poles=0,
    deg_poly=50,
    spacing="linear",
)
z_0 = 0.5 + 0.5j
r = np.sqrt(c / np.pi)
circle = lambda t: z_0 + r * np.exp(2j * np.pi * t)
circle_deriv = lambda t: 2j * np.pi * r * np.exp(2j * np.pi * t)
prob.add_interior_curve(
    lambda t: z_0 + r * np.exp(2j * np.pi * t), centroid=z_0, **params
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
# an = Analysis(sol)
# an.plot()
# plt.show()
max_error = solver.get_error()[0]
psi_top = sol.psi(1 + 1j)[0][0]
psi_bottom = sol.psi(1)[0][0]
perm_psi = (psi_top - psi_bottom) / delta_p
eps = 2 * max_error / delta_p
if np.abs(perm_psi) < 2 * eps:
    rel_error = np.inf
else:
    rel_error = np.abs(eps / (perm_psi))
print(f"{rel_error = }")
