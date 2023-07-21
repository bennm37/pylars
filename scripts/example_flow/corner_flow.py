"""Flow round a corner"""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1, 0, -1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=40, deg_poly=50
)

# 1 is the inlet, 4 is the outlet, 0,2,3,5 are the walls
# prob.add_boundary_condition("1", "p[1]", 11)
# prob.add_boundary_condition("1", "v[1]", 0)
# # outletx
# prob.add_boundary_condition("4", "p[4]", -11)
# prob.add_boundary_condition("4", "u[4]", 0)
# # walls no slip no penetration
# prob.add_boundary_condition("0", "u[0]", 0)
# prob.add_boundary_condition("0", "v[0]", 0)
# prob.add_boundary_condition("2", "u[2]", 0)
# prob.add_boundary_condition("2", "v[2]", 0)
# prob.add_boundary_condition("3", "u[3]", 0)
# prob.add_boundary_condition("3", "v[3]", 0)
# prob.add_boundary_condition("5", "u[5]", 0)
# prob.add_boundary_condition("5", "v[5]", 0)
# TODO fix the dependent in side issue
prob.name_side("1", "inlet")
prob.name_side("4", "outlet")
prob.group_sides(["0", "2", "3", "5"], "walls")
prob.add_boundary_condition("inlet", "p[inlet]", 11)
prob.add_boundary_condition("inlet", "v[inlet]", 0)
prob.add_boundary_condition("outlet", "p[outlet]", -11)
prob.add_boundary_condition("outlet", "u[outlet]", 0)
prob.add_boundary_condition("walls", "u[walls]", 0)
prob.add_boundary_condition("walls", "v[walls]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
print(f"Residual: {residual:.15e}")

# plotting
a = Analysis(sol)
fig, ax = a.plot(resolution=300)
max = a.psi_values[~np.isnan(a.psi_values)].max()
moffat_levels = max + np.linspace(-1.6e-5, 0, 10)
ax.contour(
    a.X,
    a.Y,
    a.psi_values,
    levels=moffat_levels,
    colors="y",
    linewidths=0.5,
    linestyles="solid",
)
ax.axis("off")
# plt.savefig("media/flow_round_a_corner.pdf", bbox_inches="tight")
plt.show()
