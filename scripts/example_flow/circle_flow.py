"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np

prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=600,
    num_poles=0,
    deg_poly=30,
    spacing="linear",
)
z_0 = -0.2 + 0.4j
r = 0.2
prob.add_interior_curve(
    lambda t: z_0 + r * np.exp(2j * np.pi * t),
    num_points=200,
    deg_laurent=30,
    centroid=z_0,
)
prob.domain.plot()
plt.tight_layout()
plt.savefig("media/circle_domain.pdf")


prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 25)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(sol)
fig, ax = an.plot(resolution=100, interior_patch=True, enlarge_patch=1.1)
plt.savefig("media/circle_flow.pdf")
plt.show()

from pylars.numerics import split_laurent

cf, cg, clf, clg = split_laurent(
    solver.coefficients, solver.domain.interior_laurents
)
print("Residual = ", np.abs(solver.A @ solver.coefficients - solver.b).max())
print("clf = ", clf)
print("clg = ", clg)
print("diff = ", clg + np.conj(z_0) * clf)
print("finished")
