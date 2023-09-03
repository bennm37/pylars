"""Test running and animating a LowDensityMoverSimulation."""
from pylars import Problem, SimulationAnalysis
from pylars.simulation import LowDensityMoverSimulation, Mover
from pylars.domain import generate_rv_circles
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt

porosity = 0.95
rv = lambda mu: mu
rv_args = {"mu": 0.10}
length = 2
bound = length / 2
np.random.seed(6)
centers, radii = generate_rv_circles(
    porosity=porosity,
    rv=rv,
    rv_args=rv_args,
    length=length,
    min_dist=0.1,
)
init_prob = Problem()
n_circles = len(centers)
corners = [
    bound + bound * 1j,
    -bound + bound * 1j,
    -bound - bound * 1j,
    bound - bound * 1j,
]
init_prob.add_exterior_polygon(
    corners,
    num_edge_points=50 * n_circles,
    num_poles=0,
    deg_poly=30,
    spacing="linear",
)

for centroid, radius in zip(centers, radii):
    init_prob.add_interior_curve(
        lambda t: centroid + radius * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=20,
        centroid=centroid,
        mirror_laurents=False,
        mirror_tol=bound / 2,
    )
init_prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
init_prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
init_prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
init_prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

init_prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
init_prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
init_prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
init_prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
interiors = [str(i) for i in range(4, 4 + n_circles)]
for interior in interiors:
    init_prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
    init_prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)
init_prob.domain.plot()
plt.show()
# from pylars import Analysis, Solver
# solver = Solver(init_prob)
# sol = solver.solve(weight=False,normalize=False)
# an = Analysis(sol)
# an.plot()
centroid = -0.7 + 0.6j
angle = 0.0
R = 0.05
curve = lambda t: centroid + R * np.exp(2j * np.pi * t)
deriv = lambda t: R * 2j * np.pi * np.exp(2j * np.pi * t)
cell = Mover(
    curve=curve,
    deriv=deriv,
    centroid=centroid,
    angle=angle,
)
movers = [cell]
ldms = LowDensityMoverSimulation(init_prob, movers)
results = ldms.run(0, 40, 0.2)
print(results["errors"])
an = SimulationAnalysis(results)
an.plot_pathlines(39)
plt.savefig(f"media/porous_pathlines.pdf")
plt.show()
# fig, ax, anim = an.animate_fast(
#     interval=50, resolution=200, streamline_type="starting_points"
# )
# anim.save(f"media/scaff_centroid_{centroid}.mp4")
