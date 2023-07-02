"""Analysis module for the pyls package.

Supports plotting of the contours and velocity magnitude of the solution.
"""
import numpy as np
import matplotlib.pyplot as plt
from pyls.colormaps import parula


class Analysis:
    """Class for analyzing the solution of a lighting stokes problem.

    Attributes
    ----------
    domain: Domain
    solver: Solver

    Methods
    -------
    plot():
        Plot the contours and velocity magnitude of the solution.
    """

    def __init__(self, domain, solver):
        self.domain = domain
        self.solver = solver

    def plot(self, resolution=100):
        """Plot the contours and velocity magnitude of the solution."""
        corners = self.domain.corners
        xmin, xmax = np.min(corners.real), np.max(corners.real)
        ymin, ymax = np.min(corners.imag), np.max(corners.imag)
        x, y = np.linspace(xmin, xmax, resolution), np.linspace(
            ymin, ymax, resolution
        )
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.X + 1j * self.Y
        psi, uv, p, omega = self.solver.functions
        self.Z[~self.domain.mask_contains(self.Z)] = np.nan
        self.psi_values = psi(self.Z.flatten()).reshape(resolution, resolution)
        self.uv_values = uv(self.Z.flatten()).reshape(resolution, resolution)
        fig, ax = plt.subplots()
        speed = np.abs(self.uv_values)
        pc = ax.pcolormesh(
            self.X, self.Y, speed, cmap=parula, shading="gouraud"
        )
        plt.colorbar(pc)
        ax.contour(self.X, self.Y, self.psi_values, colors="k", levels=20)
        ax.set_aspect("equal")
        return fig, ax
