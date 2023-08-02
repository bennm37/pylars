from pylars import Problem, SimulationAnalysis
from pylars.simulation import LowDensityMoverSimulation, Mover
import numpy as np
import matplotlib.pyplot as plt

init_prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
init_prob.add_exterior_polygon(
    corners,
    num_edge_points=600,
    num_poles=0,
    deg_poly=20,
    spacing="linear",
)
init_prob.add_point(-1.0 - 1.0j)
# init_prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
# init_prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
# init_prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
# init_prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

init_prob.add_boundary_condition("0", "u[0]", 0)
init_prob.add_boundary_condition("0", "v[0]", 0)
init_prob.add_boundary_condition("2", "u[2]", 0)
init_prob.add_boundary_condition("2", "v[2]", 0)
init_prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
init_prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
init_prob.add_boundary_condition("3", "p[1]-p[3][::-1]", -1)
init_prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
init_prob.add_boundary_condition("4", "p[4]", 0)
init_prob.add_boundary_condition("4", "psi[4]", 0)

centroid = 0.0 + 0.01j
angle = 0.0
velocity = 0.0 + 0.0j
angular_velocity = 0.0
R = 0.1
curve = lambda t: centroid + R * np.exp(2j * np.pi * t)
deriv = lambda t: R * 2j * np.pi * np.exp(2j * np.pi * t)
cell = Mover(
    curve=curve,
    deriv=deriv,
    centroid=centroid,
    angle=angle,
    velocity=velocity,
    angular_velocity=angular_velocity,
)
movers = [cell]
ldms = LowDensityMoverSimulation(init_prob, movers)
results = ldms.run(0, 1.5, 0.1)
an = SimulationAnalysis(results)
fig, ax, anim = an.animate(interval=100, vmax=0.4)
anim.save("media/ldms.mp4")
