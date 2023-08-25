"""poiseuille flow in a square domain.""" ""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=0, deg_poly=24, spacing="linear"
)
prob.domain.plot()
plt.show()
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "p[1]", 2)
prob.add_boundary_condition("1", "v[1]", 0)
prob.add_boundary_condition("3", "p[3]", -2)
prob.add_boundary_condition("3", "v[3]", 0)
solver = Solver(prob)
sol = solver.solve(weight=False, normalize=False)
residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
print(f"Residual: {residual:.15e}")
analysis = Analysis(sol)
fig, ax = analysis.plot()
ax.axis("off")

# plot the velocity profile on the inlet
y = np.linspace(-1, 1, 100)
ax.plot(sol.uv(1j * y - 1).real - 1, y, color="black")
y_quiver = np.linspace(-1, 1, 10)
x_quiver = -np.ones_like(y_quiver)
u = sol.uv(x_quiver + 1j * y_quiver).real
v = sol.uv(x_quiver + 1j * y_quiver).imag
ax.quiver(x_quiver, y_quiver, u, v, color="black", scale=2)
plt.savefig("media/poiseuille_flow.pdf", bbox_inches="tight")
plt.show()

# Figure 2: Plot profiles.
fig, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0, 0].plot(y, sol.uv(1j * y).real, color="black", label="solver")
ax[0, 0].plot(y, 1 - y**2, color="red", linestyle="--", label="analytical")
ax[0, 0].set(xlabel="$y$", ylabel="$u$", title="u profile")
ax[0, 0].legend()

ax[1, 0].plot(y, sol.psi(1j * y).real, color="black", label="solver")
ax[1, 0].plot(
    y, y * (1 - y**2 / 3), color="red", linestyle="--", label="analytical"
)
ax[1, 0].set_yticks(np.linspace(-0.7, 0.7, 8))
ax[1, 0].legend()
ax[1, 0].set(xlabel="$y$", ylabel="$\psi$", title="$\psi$ profile")

ax[0, 1].plot(y, sol.p(y).real, color="black", label="solver")
ax[0, 1].plot(y, -2 * y, color="red", linestyle="--", label="analytical")
ax[0, 1].legend()
ax[0, 1].set(xlabel="$x$", ylabel="$p$", title="$p$ profile")

ax[1, 1].plot(y, sol.omega(1j * y).real, color="black", label="solver")
ax[1, 1].plot(y, 2 * y, color="red", linestyle="--", label="analytical")
ax[1, 1].legend()
ax[1, 1].set(xlabel="$y$", ylabel="$\omega$", title="$\omega$ profile")
plt.tight_layout()
plt.show()
