"""Solve Poiseiulle flow with stream function boundary conditions."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# solve the tb and lr problems
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0, spacing="linear")
# dom.show()
sol_left = Solver(dom, 24)
a1, a2, a3, a4 = 1, -1, 0, 0
sol_left.add_boundary_condition("0", "u(0)", 0)
sol_left.add_boundary_condition("0", "psi(0)", a1)
sol_left.add_boundary_condition("2", "u(2)", 0)
sol_left.add_boundary_condition("2", "psi(2)", a2)
sol_left.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol_left.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
psi_left, uv_left, p_left, omega_left = sol_left.solve(
    check=False, weight=False
)
res_left = np.abs(sol_left.A @ sol_left.coefficients - sol_left.b)
print(res_left.max())
sol_top = Solver(dom, 24)
sol_top.add_boundary_condition("1", "v(1)", 0)
sol_top.add_boundary_condition("1", "psi(1)", f"y**3+{a3}")
# sol_top.add_boundary_condition("1", "psi(1)", a3)
sol_top.add_boundary_condition("3", "v(3)", 0)
sol_top.add_boundary_condition("3", "psi(3)", f"y**3+{a4}")
# sol_top.add_boundary_condition("3", "psi(3)", a4)
sol_top.add_boundary_condition("0", "u(0)-u(2)[::-1]", 0)
sol_top.add_boundary_condition("0", "v(0)-v(2)[::-1]", 0)
psi_top, uv_top, p_top, omega_top = sol_top.solve(check=False, weight=False)
res_top = np.abs(sol_top.A @ sol_top.coefficients - sol_top.b)
print(res_top.max())
# combine the solutions
a, b = 0, 1
sol_combined = a * sol_left + b * sol_top

an = Analysis(dom, sol_combined)
an.plot()
plt.show()
