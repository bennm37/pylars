"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.perf_counter()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners=corners,
    num_edge_points=600,
    num_poles=0,
    deg_poly=15,
    spacing="linear",
)
rs = [0.15, 0.15, 0.15]
cs = [0.0 + 0.0j, 0.5 + 0.5j, -0.5 - 0.5j]
deg_laurent = 10
for r, c in zip(rs, cs):
    prob.add_interior_curve(
        lambda t: c + r * np.exp(2j * np.pi * t),
        num_points=300,
        deg_laurent=deg_laurent,
    )
prob.add_boundary_condition("0", "psi[0]", 1)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("2", "psi[2]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "u[5]", 0)
prob.add_boundary_condition("5", "v[5]", 0)
prob.add_boundary_condition("6", "u[6]", 0)
prob.add_boundary_condition("6", "v[6]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
end = time.perf_counter()
print(f"Time elapsed: {end-start:.2f} seconds")
# plotting
prob.domain._update_polygon(buffer=1e-2)
an = Analysis(prob, sol)
fig, ax = an.plot(resolution=100, interior_patch=True, levels=20)
plt.show()
