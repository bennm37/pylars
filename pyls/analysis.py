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
        psi, uv, p, omega = self.solver.functions
        Z[np.logical_not(self.domain.mask_contains(Z))] = np.nan
        psi_100_100 = psi(Z.flatten()).reshape(100, 100)
        uv_100_100 = uv(Z.flatten()).reshape(100, 100)
        # plot the velocity magnitude
        fig, ax = plt.subplots()
        # interpolate using bilinear interpolation
        speed = np.abs(uv_100_100)
        pc = ax.pcolormesh(X, Y, speed, cmap="jet")
        plt.colorbar(pc)
        ax.contour(X, Y, psi_100_100, colors="k", levels=20)
        ax.set_aspect("equal")
        return fig, ax
        
