"""Test running and animating a LowDensityMoverSimulation."""
from pylars import Problem, SimulationAnalysis
from pylars.simulation import LowDensityMoverSimulation, Mover
import numpy as np

init_prob = Problem()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
init_prob.add_periodic_domain(
    2,
    2,
    num_edge_points=600,
    num_poles=0,
    deg_poly=50,
    spacing="linear",
)
init_prob.add_point(-1.0 - 1.0j)
init_prob.add_boundary_condition("0", "u[0]", 0)
init_prob.add_boundary_condition("0", "v[0]", 0)
init_prob.add_boundary_condition("2", "u[2]", 0)
init_prob.add_boundary_condition("2", "v[2]", 0)
init_prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
init_prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
init_prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
init_prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
init_prob.add_boundary_condition("4", "p[4]", 0)
init_prob.add_boundary_condition("4", "psi[4]", 0)
# init_prob.domain.plot()
# plt.show()
centroid = -0.35 + 0.2j
angle = 0.0
R = 0.6
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
results = ldms.run(0, 3.0, 0.1)
print(results["residuals"])
an = SimulationAnalysis(results)
fig, ax, anim = an.animate_fast(interval=50, streamline_type="starting_points")
anim.save(f"media/ldms_R_{R}.mp4")
