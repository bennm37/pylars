"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np

prob = Problem()
prob.add_periodic_domain(
    length=2,
    height=2,
    num_edge_points=600,
    num_poles=0,
    deg_poly=75,
    spacing="linear",
)
# corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
# prob.add_exterior_polygon(
#     corners=corners,
#     num_edge_points=600,
#     num_poles=0,
#     deg_poly=75,
#     spacing="linear",
# )
# if this is interior, then branch cut. If exterior, then no branch cut.
# prob.domain._generate_interior_laurent_series("3", 20, 1.1 + 0.1j)
prob.domain._generate_exterior_laurent_series("3", 20, 1.1 + 0.1j)
prob.domain.plot(set_lims=False)
# plt.show()
prob.add_point(-1 - 1j)
prob.add_point(1 - 1j)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
# prob.add_boundary_condition("3", "u[3]", "1-y**2")
# prob.add_boundary_condition("3", "v[3]", 0)
# prob.add_boundary_condition("1", "v[1]", 0)
# prob.add_boundary_condition("1", "p[3]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
# prob.add_boundary_condition("4", "p[4]", 1)
# prob.add_boundary_condition("4", "psi[4]", 1)
# prob.add_boundary_condition("5", "p[5]", 1)
# prob.add_boundary_condition("5", "psi[5]", 1)
solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(sol)
print(f"residual = {np.abs(solver.A @ solver.coefficients - solver.b).max()}")
fig, ax = an.plot(
    resolution=200,
    quiver=False,
    streamline_type="linear",
    n_streamlines=20,
)
plt.show()
plt.imshow(an.psi_values)
plt.show()
