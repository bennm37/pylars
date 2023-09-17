"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np

prob = Problem()
periodic = True
if not periodic:
    corners = [2 + 1j, -2 + 1j, -2 - 1j, 2 - 1j]
    prob.add_exterior_polygon(
        corners=corners,
        num_edge_points=600,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )
else:
    prob.add_periodic_domain(
        length=4,
        height=2,
        num_edge_points=600,
        num_poles=0,
        deg_poly=50,
        spacing="linear",
    )
centroid = -1.85 + 0.0j
radius = 0.1
circle = lambda t: centroid + radius * np.exp(2j * np.pi * t)  # noqa: E731
circle_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)  # noqa: E731
num_points = 100
if not periodic:
    prob.add_interior_curve(
        circle,
        num_points=num_points,
        deg_laurent=20,
        centroid=centroid,
        mirror_laurents=False,
    )
else:
    prob.add_periodic_curve(
        circle,
        num_points=num_points,
        deg_laurent=10,
        centroid=centroid,
        mirror_laurents=False,
        image_laurents=True,
    )
    prob.domain.error_points["4"] = circle(np.linspace(0, 1, 2 * num_points))
prob.domain.plot(set_lims=False)
plt.show()
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 1)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
rot_speed = 1.0 / np.pi
trans_speed = 0.0
prob.add_boundary_condition(
    "4",
    "u[4]",
    trans_speed + rot_speed * circle_deriv(np.linspace(0, 1, num_points)).real,
)
prob.add_boundary_condition(
    "4", "v[4]", rot_speed * circle_deriv(np.linspace(0, 1, num_points)).imag
)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(sol)
print(f"residusal = {np.abs(solver.A @ solver.coefficients - solver.b).max()}")
fig, ax = an.plot(
    resolution=100,
    interior_patch=True,
    quiver=False,
    streamline_type="linear",
    n_streamlines=20,
)
ax.axis("off")
fig.set_size_inches(3, 3)
plt.tight_layout()
# plt.savefig("media/rotating_flow.png", bbox_inches="tight")
plt.show()
