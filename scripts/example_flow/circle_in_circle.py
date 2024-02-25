from pylars import Problem, Solver, Analysis
from pylars.simulation import Mover
import numpy as np
import matplotlib.pyplot as plt

prob = Problem()
outer_r = 1
outer_circle = lambda t: outer_r * np.exp(2j * np.pi * t)
prob.add_curved_domain(outer_circle, num_edge_points=500, deg_poly=50)
inner_center = 0.5
inner_r = 0.2
inner_circle = lambda t: inner_center + inner_r * np.exp(2j * np.pi * t)
inner_deriv = lambda t: 2j * np.pi * inner_r * np.exp(2j * np.pi * t)
mover = Mover(
    inner_circle,
    inner_deriv,
    centroid=inner_center,
    velocity=1.0,
    angular_velocity=1,
)
prob.add_mover(mover, num_points=200, deg_laurent=30)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
solver = Solver(prob)
sol = solver.solve()
max_error, errors = solver.get_error()
print(f"Error: {max_error}")
an = Analysis(sol)
an.plot(resolution=400)
plt.show()
an.plot_errors(errors)
plt.show()
