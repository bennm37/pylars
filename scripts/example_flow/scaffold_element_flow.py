from scaffoldtk.scaffold_generator import generate_doubly_periodic_noise
from scaffoldtk.read_microCT import get_boundary_points
from matplotlib import pyplot as plt
import numpy as np

shape = (500, 500)
dp_noise = generate_doubly_periodic_noise(
    shape=shape, sigma=10, iter=1, threshold=0.505
)
plt.imshow(dp_noise, cmap="gray")
plt.show()

boundary_points = get_boundary_points(
    dp_noise, num_points=100, alpha=0.3, sort="alphashape"
)
