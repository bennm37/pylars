"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.animation as animation
import numpy as np

prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=50,
    num_poles=0,
    deg_poly=50,
    spacing="linear",
)
centroid = -0.5 + 0.4j
R = 0.2
circle = lambda t: centroid + R * np.exp(2j * np.pi * t)  # noqa: E731
cicle_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)  # noqa: E731
num_points = 50
theta = np.linspace(0, 1, num_points)
prob.add_interior_curve(
    circle,
    num_points=num_points,
    deg_laurent=5,
    centroid=centroid,
)
rho = 100.0
mass = prob.domain.area("4") * rho
moi = mass * R**2
# prob.domain.plot()
# plt.show()
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "u[3]-u[1][::-1]", 0)
prob.add_boundary_condition("1", "v[3]-v[1][::-1]", 0)
prob.add_boundary_condition("3", "p[3]-p[1][::-1]", 2)
prob.add_boundary_condition("3", "e12[3]-e12[1][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
solver = Solver(prob)
position = centroid
angle = 0.0
velocity = 0.0 + 0.0j
angular_velocity = 0.0
dt = 0.05
ts = np.arange(0, 0.8, dt)
n_steps = len(ts)
position_data = np.zeros((n_steps, 2))
velocity_data = np.zeros((n_steps, 2))
angle_data = np.zeros(n_steps)
angular_velocity_data = np.zeros(n_steps)
solutions = []
tangent = lambda t: 1j * np.exp(2j * np.pi * t)  # noqa: E731
for i, t in enumerate(ts):
    if i % 5 == 0:
        print("Computing step", i)
    sol = solver.solve(check=False, normalize=False)
    solutions += [sol]
    position_data[i] = position.real, position.imag
    velocity_data[i] = velocity.real, velocity.imag
    angle_data[i] = angle
    angular_velocity_data[i] = angular_velocity
    current_circle = lambda t: position + R * np.exp(2j * np.pi * t)
    force = sol.force(current_circle, cicle_deriv)
    torque = sol.torque(current_circle, cicle_deriv, centroid)
    acceleration = -force / mass
    angular_acceleration = torque / moi
    velocity += acceleration * dt
    angular_velocity += angular_acceleration * dt
    position += velocity * dt
    angle += angular_velocity * dt
    solver.problem.domain.translate("4", velocity * dt)
    solver.problem.domain.rotate("4", angle)
    solver.problem.boundary_conditions["4"] = [
        ("u[4]", velocity.real + angular_velocity * tangent(theta).real),
        ("v[4]", velocity.imag + angular_velocity * tangent(theta).imag),
    ]

# for i, t in enumerate(ts):
#     an = Analysis(solutions[i])
#     fig, ax = an.plot(resolution=100, interior_patch=True, buffer=1e-2)
#     ax.title.set_text(f"t = {t:.2f}")
#     plt.show()

# animating
an = Analysis(solutions[0])
vmin, vmax = 0, 0.6
fig, ax = an.plot(resolution=100, interior_patch=True, vmin=vmin, vmax=vmax)
t = 0.0
ax.set(title=f"t = {t:.2f}")
ax.quiver(
    position_data[0, 0],
    position_data[0, 1],
    velocity_data[0, 0],
    velocity_data[0, 1],
    color="k",
)
ax.quiver(
    position_data[0, 0],
    position_data[0, 1],
    np.cos(angle_data[0]),
    np.sin(angle_data[0]),
    color="r",
)


def update(i):
    """Update the animation."""
    ax.clear()
    ax.set(title=f"t = {ts[i]:.2f}")
    if i % 5 == 0:
        print("Animating frame", i)
    an = Analysis(solutions[i])
    an.plot(
        resolution=100,
        interior_patch=True,
        figax=(fig, ax),
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.title.set_text(f"t = {t:.2f}")
    ax.quiver(
        position_data[i, 0],
        position_data[i, 1],
        velocity_data[i, 0],
        velocity_data[i, 1],
        color="k",
        zorder=4,
        scale=10,
    )
    ax.quiver(
        position_data[i, 0],
        position_data[i, 1],
        np.cos(angle_data[i]),
        np.sin(angle_data[i]),
        color="r",
        zorder=3,
    )


anim = animation.FuncAnimation(fig, update, frames=n_steps, interval=75)
anim.save("media/poiseuille_oc_simulation.mp4")
