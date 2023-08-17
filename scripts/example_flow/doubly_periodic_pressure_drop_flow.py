"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=500, deg_poly=50, num_poles=0, spacing="linear"
)
# prob.add_point(-1 - 1j)
prob.add_boundary_condition("0", "psi[0]-psi[2][::1]", 1)
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)
p_drop = 0.0
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
solver = Solver(prob)
sol = solver.solve(check=False, weight=False, normalize=True)
residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
print(f"Residual: {residual:.15e}")
an = Analysis(sol)
fig, ax = an.plot(resolution=200)
plt.show()

# continuity checks
tol = 1e-5
an.plot_relative_periodicity_error(p_drop_lr=0, tol=tol)
plt.show()
an.bar_relative_periodicity_error(p_drop_lr=0, tol=tol)
plt.show()
