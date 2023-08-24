"""Create labelled polygonal domains."""
from pylars import Problem
import numpy as np
import matplotlib.pyplot as plt

# Create a square domain
prob = Problem()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob.add_exterior_polygon(corners)
prob.domain.plot(point_color="k")
plt.show()

# Create a near touching geometry
prob = Problem()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob.add_exterior_polygon(corners, num_poles=0, spacing="linear")
circle = lambda t: 0.79 + 0.2 * np.exp(2j * np.pi * t)
prob.add_interior_curve(circle, mirror_laurents=True, mirror_tol=0.5)
prob.domain.plot(point_color="k", set_lims=False)
plt.show()

# Create an irregular shaped blob in a circle
