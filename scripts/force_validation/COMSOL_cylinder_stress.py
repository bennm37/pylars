"""Compare stress data to COMSOL."""
import pandas as pd
import matplotlib.pyplot as plt
import time
from pylars import Problem, Solver, Analysis
import numpy as np

plt.style.use("ggplot")
start = time.perf_counter()
prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=600,
    num_poles=0,
    deg_poly=20,
    spacing="linear",
)
prob.add_interior_curve(
    lambda t: 0.5 * np.exp(2j * np.pi * t),
    num_points=100,
    deg_laurent=20,
    centroid=0.0 + 0.0j,
)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
# prob.add_boundary_condition("3", "u[3]", "1-y**2")
# prob.add_boundary_condition("3", "v[3]", 0)
# prob.add_boundary_condition("1", "p[1]", 0)
# prob.add_boundary_condition("1", "v[1]", 0)
prob.add_boundary_condition("3", "p[3]", 1)
prob.add_boundary_condition("3", "v[3]", 0)
prob.add_boundary_condition("1", "p[1]", -1)
prob.add_boundary_condition("1", "v[1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
# prob.domain.plot()
# plt.show()
solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
end = time.perf_counter()
print(str(end - start) + " seconds")
an = Analysis(sol)
fig, ax = an.plot(resolution=200, interior_patch=True, enlarge_patch=1.1)
plt.savefig("media/circle_flow.pdf")
plt.show()

# calculate normal stress data
stress_x_df = pd.read_csv("tests/data/COMSOL_cylinder_stress_x.csv")
theta = stress_x_df["theta"]
circle = lambda t: 0.5 * np.exp(t * 1j)  # noqa E731
normal = lambda t: -np.exp(t * 1j)  # noqa E731
z = circle(np.array(theta))
stress = sol.stress_goursat(z)
normals = normal(np.array(theta))
normals = np.array([normals.real, normals.imag]).T
# normal_stress = np.einsum('ij,ljk->ij', normals, stress)
normal_stress = np.array([S @ n for n, S in zip(normals, stress)])

# normal stress data from COMSOL
stress_x_df = pd.read_csv("tests/data/COMSOL_cylinder_stress_x.csv")
stress_y_df = pd.read_csv("tests/data/COMSOL_cylinder_stress_y.csv")
theta = stress_x_df["theta"]
stress_x = stress_x_df["stress_x"]
stress_y = stress_y_df["stress_y"]
fig, ax = plt.subplots(1, 2)
sample = 1
sf = 1
ax[0].plot(theta[::sample], stress_x[::sample], label="COMSOL Stress X")
ax[0].plot(theta[::sample], stress_y[::sample], label="COMSOL Stress Y")
ax[0].plot(
    theta,
    normal_stress[:, 0] / sf,
    linestyle="--",
    c="k",
    label="PyLARS Stress X",
)
ax[0].plot(
    theta,
    normal_stress[:, 1] / sf,
    linestyle="--",
    c="darkgreen",
    label="PyLARS Stress Y",
)
ax[0].set(
    xlabel=" $\Theta$ ",
    ylabel="Normal Stress Components",
    title="Normal Stress on Cylinder",
)
ax[0].legend()
sample = 15
offset = 7
ax[1].plot(
    theta[offset::sample],
    stress_x[offset::sample] - normal_stress[offset::sample, 0] / sf,
    label="Error X",
)
ax[1].plot(
    theta[offset::sample],
    stress_y[offset::sample] - normal_stress[offset::sample, 1] / sf,
    label="Error Y",
)
ax[1].set(
    xlabel=" $\Theta$ ",
    ylabel="Error",
    title="Error in Normal Stress on Cylinder",
)
ax[1].legend()
# plt.savefig("media/COMSOL_vs_PyLARS_stress.pdf")
plt.show()
