from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

c = 0.76
R = np.sqrt(c / np.pi)
deg = 50
params = {
    "num_points": 800,
    "deg_laurent": deg,
    "mirror_laurents": False,
    "mirror_tol": 1.5,
}
prob = Problem()
corners = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=800,
    num_poles=0,
    deg_poly=20,
    spacing="linear",
)
z_0 = 0.5 + 0.5j
r = np.sqrt(c / np.pi)
circle = lambda t: z_0 + r * np.exp(2j * np.pi * t)
circle_deriv = lambda t: 2j * np.pi * r * np.exp(2j * np.pi * t)
prob.add_interior_curve(
    lambda t: z_0 + r * np.exp(2j * np.pi * t), centroid=z_0, **params
)
# Experimental
# circle laurent series
w = 0.30
prob.domain._generate_interior_laurent_series(1, deg, z_0 + w)
prob.domain._generate_interior_laurent_series(1, deg, z_0 - w)
prob.domain._generate_interior_laurent_series(1, deg, z_0 + w * 1j)
prob.domain._generate_interior_laurent_series(1, deg, z_0 - w * 1j)
# mirrored laurent series
w2 = 1
prob.domain._generate_exterior_laurent_series(1, deg, z_0 + w2)
prob.domain._generate_exterior_laurent_series(1, deg, z_0 - w2)
prob.domain._generate_exterior_laurent_series(1, deg, z_0 + w2 * 1j)
prob.domain._generate_exterior_laurent_series(1, deg, z_0 - w2 * 1j)
# in circle outward clustered
# UD
# s=0.9
# prob.domain._generate_clustered_poles(
#     10, 0.5 + 1j * (0.5 + s * R), -1j, length_scale=0.1
# )
# prob.domain._generate_clustered_poles(
#     10, 0.5 + 1j * (0.5 - s * R), 1j, length_scale=0.1
# )
# # LR
# prob.domain._generate_clustered_poles(10, (0.5 + s * R) + 0.5j, -1, length_scale=0.1)
# prob.domain._generate_clustered_poles(10, (0.5 - s * R) + 0.5j, 1, length_scale=0.1)
# ring
# ring_radius = 2
# prob.domain._generate_pole_ring(100, radius=1, center=0.5 + 0.5j)
# out circle inward clustered
# hp = 0.1
# prob.domain._generate_clustered_poles(10, 0.5 + (1 + hp) * 1j, -1j, length_scale=1)
# prob.domain._generate_clustered_poles(10, 0.5 - hp * 1j, 1j, length_scale=1)
# exterior laurents
# h = 0.5
# d = 50
# prob.domain._generate_exterior_laurent_series(0, d, 0.5 + 1j * (1 + h))
# prob.domain._generate_exterior_laurent_series(0, d, 0.5 - 1j * h)
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
