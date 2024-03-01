from scaffoldtk.scaffold_generator import generate_noise
from scaffoldtk.read_microCT import get_boundary_curves, evaluate_curve
from pylars import Problem, Solver, Analysis
from matplotlib import pyplot as plt
import numpy as np

# generating the blobs
np.random.seed(0)
shape = (500, 500)
dp_noise = generate_noise(shape=shape, sigma=15, iter=1, porosity=0.95, clear=True)
num_points = 100
tcks = get_boundary_curves(dp_noise, 0.00)


def comp_evaluate_curve(t, tck):
    """Complex wrapper for evaluate curve."""
    result = evaluate_curve(t, tck)
    result = 2 * result / 500 - 1
    if isinstance(t, np.ndarray):
        return result[:, 0] + 1j * result[:, 1]
    else:
        return result[0] + 1j * result[1]


t = np.linspace(0, 1, 200)
plt.imshow(dp_noise.T)
for tck in tcks:
    points = evaluate_curve(t, tck)
    plt.plot(points[:, 0], points[:, 1])
plt.show()

prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
prob.add_periodic_domain(
    2,
    2,
    num_edge_points=1500,
    num_poles=0,
    deg_poly=100,
    spacing="linear",
)

n_curves = len(tcks)
for tck in tcks:
    prob.add_periodic_curve(
        lambda t: comp_evaluate_curve(t, tck),
        num_points=600,
        deg_laurent=50,
        aaa=True,
        aaa_mmax=50,
    )
prob.domain.plot(set_lims=False)
plt.show()
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 15)
interiors = [str(i) for i in range(4, 4 + n_curves)]
for interior in interiors:
    prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
    prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)

solver = Solver(prob, verbose=True)
sol = solver.solve(check=False, normalize=False, weight=False)
an = Analysis(sol)
print(f"Residual: {np.abs(solver.A @ solver.coefficients - solver.b).max():.2e}")
fig, ax = an.plot(resolution=200, interior_patch=True, enlarge_patch=1.1)
fig.set_size_inches(3, 3)
ax.axis("off")
plt.tight_layout()
plt.savefig("media/scaffold_flow.png", bbox_inches="tight")
plt.show()
