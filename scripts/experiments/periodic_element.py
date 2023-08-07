from pylars import PeriodicDomain
import matplotlib.pyplot as plt
import numpy as np

dom = PeriodicDomain(2, 2)
R = 0.5
centroid = 0.75 + 0.75j
circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
dom.add_periodic_curve(circle, centroid)
fig, ax = dom.plot()
plt.show()
