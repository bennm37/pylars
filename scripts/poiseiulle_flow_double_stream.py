"""Solve Poiseiulle flow with stream function boundary conditions."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0, spacing="linear")
sol_left = Solver(dom, 24)
sol_left.add_boundary_condition("0", "u(0)", 0)
sol_left.add_boundary_condition("0", "psi(0)", 1)
sol_left.add_boundary_condition("2", "u(2)", 0)
sol_left.add_boundary_condition("2", "psi(2)", -1)
sol_left.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol_left.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
psi_left, uv_left, p_left, omega_left = sol_left.solve(
    check=False, weight=False
)
sol_top = Solver(dom, 24)
sol_top.add_boundary_condition("1", "v(1)", 0)
sol_top.add_boundary_condition("1", "psi(1)", 1)
sol_top.add_boundary_condition("3", "v(3)", 0)
sol_top.add_boundary_condition("3", "psi(3)", -1)
sol_top.add_boundary_condition("0", "u(0)-u(2)[::-1]", 0)
sol_top.add_boundary_condition("0", "v(0)-v(2)[::-1]", 0)
psi_top, uv_top, p_top, omega_top = sol_top.solve(check=False, weight=False)
sol_combined = Solver(dom, 24)
a, b = 1, 3
psi_combined = lambda z: a * psi_left(z) + b * psi_top(z)
uv_combined = lambda z: a * uv_left(z) + b * uv_top(z)
p_combined = lambda z: a * p_left(z) + b * p_top(z)
omega_combined = lambda z: a * omega_left(z) + b * omega_top(z)
sol_combined.functions = [
    psi_combined,
    uv_combined,
    p_combined,
    omega_combined,
]
an = Analysis(dom, sol_combined)
# assert the flow is doubly periodic
boundary_top = dom.boundary_points[dom.indices["0"]]
boundary_bottom = dom.boundary_points[dom.indices["2"]]
boundary_left = dom.boundary_points[dom.indices["1"]]
boundary_right = dom.boundary_points[dom.indices["3"]]
assert np.allclose(
    uv_combined(boundary_top), uv_combined(boundary_bottom[::-1])
)
assert np.allclose(
    uv_combined(boundary_left), uv_combined(boundary_right[::-1])
)
print("Flow is doubly periodic")
#  print psi, u and v at the corners
print("U at corners: ", uv_combined(corners).real)
print("V at corners: ", uv_combined(corners).imag)
print("Psi at corners: ", psi_combined(corners))
fig, ax = an.plot_periodic()
ax.axis("off")
plt.show()
