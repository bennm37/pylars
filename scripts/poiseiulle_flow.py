from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0, spacing="linear")
dom.show()
sol = Solver(dom, 24, weight_flag=False)
sol.add_boundary_condition("0", "u(0)", 0)
sol.add_boundary_condition("0", "v(0)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "v(2)", 0)
# parabolic inlet
sol.add_boundary_condition("1", "u(1)", "1 - y**2")
sol.add_boundary_condition("1", "v(1)", 0)
sol.add_boundary_condition("3", "p(3)", 0)
sol.add_boundary_condition("3", "v(3)", 0)
psi, uv, p, omega = sol.solve()

residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
analysis = Analysis(dom, sol)
analysis.plot()
plt.show()