"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.interpolate import griddata

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def fmt(x, pos):
    return f"{x:.1e}"


# create a square domain
shift = 0.0 + 0.0j
corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
corners += shift
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=200, deg_poly=50, num_poles=0, spacing="linear"
)
centroid = shift + 0.5 - 0.5j
R = 0.2j
prob.add_interior_curve(
    lambda t: centroid + R * np.exp(2j * np.pi * t),
    num_points=50,
    deg_laurent=10,
    centroid=centroid,
)
centroid2 = shift + 0.5j
R = 0.2j
prob.add_interior_curve(
    lambda t: centroid2 + R * np.exp(2j * np.pi * t),
    num_points=50,
    deg_laurent=10,
    centroid=centroid2,
)
prob.add_point(shift + -1 - 1j)
# prob.domain.plot()
# plt.show()

# top and bottom periodicity
p_drop = 2.0
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

prob.add_boundary_condition("1", "psi[1]-psi[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "u[5]", 0)
prob.add_boundary_condition("5", "v[5]", 0)
prob.add_boundary_condition("6", "p[6]", 0)
prob.add_boundary_condition("6", "psi[6]", 0)

solver = Solver(prob, verbose=True)
sol = solver.solve(check=False, weight=False, normalize=False)
print(f"Error: {solver.max_error}")
# sol.problem.domain._update_polygon(buffer=1e-5)
an = Analysis(sol)
an.plot(resolution=300)
plt.show()
fig, ax = an.plot_periodic(
    interior_patch=True, resolution=200, n_streamlines=50
)
plt.show()
plt.savefig("media/doubly_periodic_pressure_drop_flow_object.pdf")

# plotting error
an.plot_errors(solver.errors)
plt.show()

# compare to COMSOL data
CS_p_data = pd.read_csv("data/dp_two_circles/pressure.txt")
CS_v_data = pd.read_csv("data/dp_two_circles/velocity.txt")
CS_s_data = pd.read_csv("data/dp_two_circles/velocity_magnitude.txt")
CS_p_x, CS_p_y = CS_p_data["x"].to_numpy(), CS_p_data["y"].to_numpy()
CS_p_z = CS_p_x + 1j * CS_p_y
CS_p = CS_p_data["Pressure"].to_numpy()
CS_v_x, CS_v_y = CS_v_data["x"].to_numpy(), CS_v_data["y"].to_numpy()
CS_v_z = CS_v_x + 1j * CS_v_y
CS_u, CS_v = CS_v_data["u"].to_numpy(), CS_v_data["v"].to_numpy()
CS_s_x, CS_s_y = CS_s_data["x"].to_numpy(), CS_s_data["y"].to_numpy()
CS_s_z = CS_s_x + 1j * CS_s_y
CS_s = CS_s_data["Velocity magnitude"].to_numpy()


n_samples = 200
fig, ax = plt.subplots(1, 2)
# plotting pressure
print("Plotting P")
pylars_p = sol.p(CS_p_z).reshape(-1, 1)
p_X, p_Y = np.meshgrid(
    np.linspace(-1, 1, n_samples), np.linspace(-1, 1, n_samples)
)
p_Z = p_X + 1j * p_Y
mask = prob.domain.mask_contains(p_Z)
# mask P_X, P_Y using mask_contains
p_diff = np.abs(pylars_p - CS_p.reshape(-1, 1))
gd = griddata(np.array([CS_p_x, CS_p_y]).T, p_diff, (p_X, p_Y), method="cubic")
gd = gd.reshape(n_samples, n_samples)
gd[~mask] = np.nan
im = ax[0].imshow(
    gd,
    extent=[-1, 1, -1, 1],
    origin="lower",
    alpha=0.4,
    norm=LogNorm(vmin=np.min(p_diff), vmax=np.max(p_diff)),
)
scatter = ax[0].scatter(CS_p_x, CS_p_y, c=p_diff, s=1)
plt.colorbar(
    scatter, fraction=0.046, pad=0.04, format=ticker.FuncFormatter(fmt)
)
ax[0].set_aspect("equal")
ax[0].axis("off")
ax[0].set_title("Difference in Pressure")

# plotting difference in U
print("Plotting U")
pylars_s = np.abs(sol.uv(CS_s_z)).reshape(-1, 1)
p_X, p_Y = np.meshgrid(
    np.linspace(-1, 1, n_samples), np.linspace(-1, 1, n_samples)
)
p_Z = p_X + 1j * p_Y
mask = prob.domain.mask_contains(p_Z)
# mask P_X, P_Y using mask_contains
s_diff = np.abs(pylars_s - CS_s.reshape(-1, 1))
gd = griddata(np.array([CS_s_x, CS_s_y]).T, s_diff, (p_X, p_Y), method="cubic")
gd = gd.reshape(n_samples, n_samples)
gd[~mask] = np.nan
im = ax[1].imshow(
    gd,
    extent=[-1, 1, -1, 1],
    origin="lower",
    alpha=0.4,
    norm=LogNorm(vmin=np.min(s_diff), vmax=np.max(s_diff)),
)
scatter = ax[1].scatter(CS_s_x, CS_s_y, c=s_diff, s=1)
plt.colorbar(
    scatter, fraction=0.046, pad=0.04, format=ticker.FuncFormatter(fmt)
)
ax[1].set_aspect("equal")
ax[1].axis("off")
ax[1].set_title("Difference in Velocity Magnitude")
plt.tight_layout()
plt.savefig("media/dp_two_circles_flow.pgf", bbox_inches="tight")