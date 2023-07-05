"""Solve Poiseiulle flow with stream function boundary conditions."""
from pyls import Domain, Solver, Analysis
import time
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0, spacing="linear")
sol_left = Solver(dom, 24)
a1, a2, a3, a4 = 1, -1, 1, -1
sol_left.add_boundary_condition("0", "u(0)", 0)
sol_left.add_boundary_condition("0", "psi(0)", a1)
sol_left.add_boundary_condition("2", "u(2)", 0)
sol_left.add_boundary_condition("2", "psi(2)", a2)
sol_left.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol_left.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
psi_left, uv_left, p_left, omega_left = sol_left.solve(
    check=False, weight=False
)
sol_top = Solver(dom, 24)
sol_top.add_boundary_condition("1", "v(1)", 0)
sol_top.add_boundary_condition("1", "psi(1)", f"(y - 1)**2 * (y + 1)**2+{a3}")
# sol_top.add_boundary_condition("1", "psi(1)", a3)
sol_top.add_boundary_condition("3", "v(3)", 0)
sol_top.add_boundary_condition("3", "psi(3)", f"(y - 1)**2 * (y + 1)**2+{a4}")
# sol_top.add_boundary_condition("3", "psi(3)", a4)
sol_top.add_boundary_condition("0", "u(0)-u(2)[::-1]", 0)
sol_top.add_boundary_condition("0", "v(0)-v(2)[::-1]", 0)
psi_top, uv_top, p_top, omega_top = sol_top.solve(check=False, weight=False)


# combine the solutions
# a, b = 1, 1
# sol_combined = a * sol_left + b * sol_top
# psi_combined, uv_combined, p_combined, omega_combined = sol_combined.functions

# # check the flow
# boundary_top = dom.boundary_points[dom.indices["0"]]
# boundary_bottom = dom.boundary_points[dom.indices["2"]]
# boundary_left = dom.boundary_points[dom.indices["1"]]
# boundary_right = dom.boundary_points[dom.indices["3"]]
# assert np.allclose(
#     uv_combined(boundary_top), uv_combined(boundary_bottom[::-1])
# )
# assert np.allclose(
#     uv_combined(boundary_left), uv_combined(boundary_right[::-1])
# )
# print("Flow is doubly periodic")
# #  print psi, u and v at the corners
# print("U at corners: ", uv_combined(corners).real)
# print("V at corners: ", uv_combined(corners).imag)
# print("Psi at corners: ", psi_combined(corners))
theta = np.linspace(0, np.pi, 100)
a_values = np.cos(theta)
b_values = np.sin(theta)
an = Analysis(dom, sol_left)
fig, ax, anim = an.animate_combination(sol_top, a_values, b_values, n_tile=3)
anim.save("media/double_stream_circle.mp4", fps=20)
