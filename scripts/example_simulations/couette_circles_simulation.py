"""Test running and animating a LowDensityMoverSimulation."""
from pylars import Problem, SimulationAnalysis
from pylars.simulation import LowDensityMoverSimulation, Mover
from pylars.domain import generate_rv_circles
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt

length = 2
bound = length / 2
init_prob = Problem()
prob = Problem()
outer_circle = lambda t: np.exp(2j * np.pi * t)
init_prob.add_curved_domain(
    outer_circle,
    num_edge_points=500,
    deg_poly=50,
    spacing="linear",
)

centroid, radius = 0.0 + 0.0j, 0.4
inner_circle = lambda t: centroid + radius * np.exp(2j * np.pi * t)
inner_deriv = lambda t: radius * 2j * np.pi * np.exp(2j * np.pi * t)
rotating_inner = Mover(
    inner_circle, inner_deriv, centroid, velocity=0.0, angular_velocity=1.0
)
init_prob.add_mover(rotating_inner, num_points=300, deg_laurent=30)

init_prob.add_boundary_condition("0", "u[0]", 0)
init_prob.add_boundary_condition("0", "v[0]", 0)
init_prob.domain.plot()
# from pylars import Analysis, Solver
# solver = Solver(init_prob)
# sol = solver.solve(weight=False, normalize=False)
# an = Analysis(sol)
# an.plot()
# plt.show()
centroid = -0.65 + 0.0j
angle = 0.0
R = 0.2
curve = lambda t: centroid + R * np.exp(2j * np.pi * t)
deriv = lambda t: R * 2j * np.pi * np.exp(2j * np.pi * t)
cell = Mover(
    curve=curve,
    deriv=deriv,
    centroid=centroid,
    angle=angle,
)
movers = [cell]
ldms = LowDensityMoverSimulation(
    init_prob, movers, num_points=200, deg_laurent=30
)
results = ldms.run(0, 20, 0.2)
print(results["errors"])
an = SimulationAnalysis(results)
an.plot_pathlines(-1)
plt.savefig(f"media/couette_pathlines.pdf")
plt.show()
# fig, ax, anim = an.animate_fast(
#     interval=50, resolution=200, streamline_type="starting_points"
# )
# anim.save(f"media/scaff_centroid_{centroid}.mp4")
