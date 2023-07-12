"""Create labelled polygonal domains."""
from pylars import Domain
import numpy as np

# Create a square domain
corners = [0, 1, 1 + 1j, 1j]
domain = Domain(corners)
domain.show()

# Create a regular hexagonal domain
corners = [np.exp(2j * np.pi * i / 6) for i in range(6)]
domain = Domain(corners, spacing="linear")
domain.show()

# Create a non-convex domain
corners = [0, 1, 1 + 1j, 0.5 + 0.5j, 0.5j]
domain = Domain(corners, num_edge_points=300, num_poles=0)
domain.show()
