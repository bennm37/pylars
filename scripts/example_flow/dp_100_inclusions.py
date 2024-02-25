"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt
from circle_flow_2 import generate_normal_circles
import time

# create a square domain
# k = 100
# deg_poly = k
# num_edge_points = 4 * k
# deg_laurent = int(0.2 * k)
# num_points = int(0.5 * k)
num_edge_points = 600
deg_poly = 50
num_points = 150
deg_laurent = 10
print(f"deg_poly: {deg_poly}, deg_laurent: {deg_laurent}")
corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
prob = Problem()
prob.add_periodic_domain(
    length=2,
    height=2,
    num_edge_points=num_edge_points,
    deg_poly=deg_poly,
    num_poles=0,
    spacing="linear",
)
n_circles = 100
np.random.seed(11)
shift = -0.0
centroids, radii = generate_normal_circles(n_circles, 0.02, 0.00)

# see if A tall skinny enough (conservative estimate)
height = 2 * (n_circles * num_points + 4 * num_edge_points)
width = 4 * (deg_laurent * (1.5 * n_circles) + n_circles + deg_poly + 1)
print(f"height: {height}, width: {width}")
if height < 4 * width:
    raise ValueError("A is not tall skinny enough.")
# centroids = [0.80356341 + 0.36518719j, 0.336139 - 0.48264786j]
# centroids = [(-0.5 + 0.5j), (0.5 - 0.49j)]
# radii = [0.2, 0.2]
print("Circles generated")
for centroid, radius in zip(centroids, radii):
    centroid += shift
    prob.add_periodic_curve(
        lambda t: centroid + radius * np.exp(2j * np.pi * t),
        num_points=num_points,
        deg_laurent=deg_laurent,
        centroid=centroid,
        image_laurents=True,
        image_tol=0.3,
        mirror_laurents=False,
        mirror_tol=0.5,
    )
prob.add_point(-1 - 1j)
prob.domain.plot(set_lims=False)
plt.show()

# top and bottom periodicity
p_drop = 120
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
interiors = [str(i) for i in range(4, 4 + n_circles)]
for interior in interiors:
    prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0.0)
    prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)
prob.add_boundary_condition(f"{4 + n_circles}", f"p[{4 + n_circles}]", 0)
prob.add_boundary_condition(f"{4 + n_circles}", f"psi[{4 + n_circles}]", 0)
start = time.perf_counter()
solver = Solver(prob, verbose=True)
print("Solving the problem")
sol = solver.solve(check=False, weight=False, normalize=False)
print(f"Error: {solver.max_error}")
end = time.perf_counter()
print(f"Time taken: {end-start:.2f}s")
an = Analysis(sol)
fig, ax = an.plot_periodic(interior_patch=True, enlarge_patch=1.01)
fig.set_size_inches(3, 3)
ax.axis("off")
plt.tight_layout()
plt.savefig("media/dp_100_inclusions.png", bbox_inches="tight")
an.plot_relative_periodicity_error()
plt.show()
an.bar_plot_relative_periodicity_error()
plt.show()
