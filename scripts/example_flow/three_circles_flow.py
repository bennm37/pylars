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
rs = [0.3, 0.3, 0.3]
cs = [0.0 + 0.0j, -0.5 - 0.5j, +0.5 + 0.5j]
degrees = [10, 10, 10]
for r, c, deg in zip(rs, cs, degrees):
    prob.add_interior_curve(
        lambda t: c + r * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=deg,
    )
prob.domain.plot()
plt.show()
