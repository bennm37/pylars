"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem
import matplotlib.pyplot as plt
import numpy as np

corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners=corners,
    num_edge_points=300,
    num_poles=10,
    deg_poly=24,
    spacing="linear",
)
rs = [0.2, 0.2, 0.2]
cs = [0.0 + 0.0j, -0.5 - 0.5j, +0.5 + 0.5j]
degrees = [10, 20, 10]
for r, c, deg in zip(rs, cs, degrees):
    prob.add_interior_curve(
        lambda t: c + r * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=deg,
        centroid=c,
    )
# prob.name_side("1", "inlet")
# prob.name_side("3", "outlet")
# prob.group_sides(["0","2"],"walls")
# prob.group_sides(["4","5","6"],"scaffold")
# prob.add_boundary_condition("inlet", "u[inlet]-u[outlet]", 0)
# prob.add_boundary_condition("inlet", "v[inlet]-v[outlet]", 0)
# prob.add_boundary_condition("walls", "u[walls]", 0)
# prob.add_boundary_condition("walls", "v[walls]", 0)
# prob.add_boundary_condition("scaffold","u[scaffold]", 0)
# prob.add_boundary_condition("scaffold","v[scaffold]", 0)
prob.domain.plot()
plt.show()
