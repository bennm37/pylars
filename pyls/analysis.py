import numpy as np
import matplotlib.pyplot as plt
import shapely


class Analysis:
    def __init__(self, domain, solver):
        self.domain = domain
        self.solver = solver

    def plot(self):
        # get bounding box of domain
        corners = self.domain.corners
        xmin, xmax = np.min(corners.real), np.max(corners.real)
        ymin, ymax = np.min(corners.imag), np.max(corners.imag)
        x, y = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        mask = self.domain.mask_contains(Z)
        Z[mask] = np.nan
        
