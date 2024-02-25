"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
from pylars.domain.generation import rectangle, polygon
import matplotlib.pyplot as plt
import numpy as np

prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
# prob.add_exterior_polygon(
#     corners,
#     num_edge_points=600,
#     num_poles=0,
#     deg_poly=30,
#     spacing="linear",
# )
prob.add_periodic_domain(
    2,
    2,
    num_edge_points=1000,
    num_poles=0,
    deg_poly=100,
    spacing="linear",
)

z_0, L = -0 - 0j, 0.4
prob.add_periodic_curve(
    rectangle(z_0, L, L),
    num_points=500,
    deg_laurent=100,
    # mirror_laurents=True,
    # mirror_tol=1,
    # image_laurents=True,
    # image_tol=1,
    aaa=True,
    centroid=z_0,
)
# prob.domain.plot()
# plt.show()

prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 1)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)


solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(sol)
fig, ax = an.plot(
    resolution=100,
    interior_patch=True,
    n_streamlines=21,
    streamline_type="starting_points",
    enlarge_patch=1.1,
)
plt.show()
