"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(corners, num_edge_points=300, num_poles=0, spacing="linear")
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 4)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
solver = Solver(prob)
sol = solver.solve(check=True, weight=False, normalize=False)
print(f"Error: {solver.max_error}")
a = Analysis(sol)
fig, ax = a.plot()
plt.show()
