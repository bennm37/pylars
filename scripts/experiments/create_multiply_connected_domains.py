"""Create labelled polygonal domains."""

from pylars import Problem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

foldername = "media"
# Create a near touching geometry
prob = Problem()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob.add_exterior_polygon(corners, num_poles=10, spacing="linear")
circle = lambda t: 0.78 + 0.2 * np.exp(2j * np.pi * t)
prob.add_interior_curve(circle, mirror_laurents=True, mirror_tol=0.5)
prob.domain.plot(point_color="k", set_lims=False)
plt.savefig(
    f"{foldername}/touching_domain" + ".png",
    bbox_inches="tight",
)
plt.show()

# Create an irregular shaped blob in a circle
prob = Problem()
outer_circle = lambda t: np.exp(2j * np.pi * t)
prob.add_curved_domain(outer_circle, num_edge_points=500)
cardioid = lambda t: 0.5 + 0.1 * (1 - (np.exp(2j * np.pi * t) - 1) ** 2)
prob.add_interior_curve(cardioid, num_points=300, aaa=True)
prob.domain.plot()
# plt.savefig(
# f"{foldername}/curved_domain" + ".png", backend="png", bbox_inches="tight"
# )
plt.show()

# Create a periodic domain with over boundary objects
prob = Problem()
prob.add_periodic_domain(2, 2, num_edge_points=300)
R = 0.5
centroid = 0.75 + 0.75j
circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
prob.add_periodic_curve(circle, centroid, num_points=200)
prob.domain.plot()
plt.savefig(
    f"{foldername}/periodic_domain" + ".png",
    bbox_inches="tight",
)
plt.show()
