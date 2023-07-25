"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np


def generate_circles(n_circles, radius):
    """Generate non-overlapping circles."""
    L = 1.9
    centroids = np.array(
        L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
    )
    n_current = 1
    i = 0
    while n_current < n_circles:
        i += 1
        if i % 100000 == 0:
            print(i)
            print(f"{n_current=}")
        centroid = (
            L * np.random.rand(1)
            - L / 2
            + 1j * (L * np.random.rand(1) - L / 2)
        )
        if np.min(np.abs(centroid - centroids)) > 2 * radius:
            centroids = np.append(centroids, centroid)
            n_current += 1
    return centroids


def generate_normal_circles(n_circles, mean, std):
    """Generate non-overlapping circles."""
    L = 2 - 1.5 * (mean + 6 * std)
    radii = np.array(np.random.normal(mean, std, 1))
    centroids = np.array(
        L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
    )
    n_current = 1
    radius = np.random.normal(mean, std, 1)
    while n_current < n_circles:
        centroid = (
            L * np.random.rand(1)
            - L / 2
            + 1j * (L * np.random.rand(1) - L / 2)
        )
        if np.min(np.abs(centroid - centroids) / (radii + radius)) > 1:
            centroids = np.append(centroids, centroid)
            radii = np.append(radii, radius)
            n_current += 1
            radius = np.random.normal(mean, std, 1)
    return centroids, radii


prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=100,
    num_poles=0,
    deg_poly=100,
    spacing="linear",
)
# centroids = [0.4 + 0.5j, 0.5 - 0.6j, -0.5 - 0.5j, -0.5 + 0.5j]
n_circles = 50
centroids, radii = generate_normal_circles(n_circles, 0.03, 0.01)
print("Circles generated")
for centroid, radius in zip(centroids, radii):
    prob.add_interior_curve(
        lambda t: centroid + radius * np.exp(2j * np.pi * t),
        num_points=50,
        deg_laurent=10,
        centroid=centroid,
    )

prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "psi[0]", -2 / 3)
# prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
# prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("2", "psi[2]", 2 / 3)
prob.add_boundary_condition("1", "u[1]", "1-y**2")
prob.add_boundary_condition("1", "v[1]", 0)
prob.add_boundary_condition("3", "p[3]", 0)
prob.add_boundary_condition("3", "v[3]", 0)
interiors = [str(i) for i in range(4, 4 + n_circles)]
for interior in interiors:
    prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
    prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False, weight=False)
an = Analysis(sol)
sol.problem.domain.enlarge_holes(1.0)
print(f"Residual: {np.abs(solver.A @ solver.coefficients - solver.b).max()}")
fig, ax = an.plot(resolution=200, interior_patch=True, epsilon=0.01)
plt.show()
