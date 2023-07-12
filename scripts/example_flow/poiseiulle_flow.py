"""Poiseiulle flow in a square domain.""" ""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=0, deg_poly=24, spacing="linear"
)
prob.domain.show()
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
# parabolic inlet
prob.add_boundary_condition("1", "p[1]", 2)
prob.add_boundary_condition("1", "v[1]", 0)
# outlet
prob.add_boundary_condition("3", "p[3]", -2)
prob.add_boundary_condition("3", "v[3]", 0)
solver = Solver(prob)
sol = solver.solve(weight=False, normalize=False)

residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
print(f"Residual: {residual:.15e}")
analysis = Analysis(prob, sol)
fig, ax = analysis.plot()
ax.axis("off")
# plot the velocity profile on the inlet
y = np.linspace(-1, 1, 100)
ax.plot(sol.uv(1j * y - 1).real - 1, y, color="black")
y = np.linspace(-1, 1, 10)
x = -np.ones_like(y)
u = sol.uv(x + 1j * y).real
v = sol.uv(x + 1j * y).imag
ax.quiver(x, y, u, v, color="black", scale=2)
plt.savefig("media/poiseiulle_flow.pdf", bbox_inches="tight")
plt.show()
