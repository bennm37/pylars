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
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.Z = self.X + 1j * self.Y
        psi, uv, p, omega = self.solver.functions
        self.Z[~self.domain.mask_contains(self.Z)] = np.nan
        self.psi_values = psi(self.Z.flatten()).reshape(resolution, resolution)
        self.uv_values = uv(self.Z.flatten()).reshape(resolution, resolution)
        fig, ax = plt.subplots()
        speed = np.abs(self.uv_values)
        parula.set_bad("white")
        pc = ax.pcolormesh(
            self.X, self.Y, speed, cmap=parula, shading="gouraud"
        )
        plt.colorbar(pc)
        ax.contour(
            self.X,
            self.Y,
            self.psi_values,
            colors="k",
            levels=10,
            linestyles="solid",
            linewidths=0.5,
        )
        ax.set_aspect("equal")
        return fig, ax

    def plot_periodic(self, resolution=100):
        """Plot the contours and velocity magnitude of the solution."""
        corners = self.domain.corners
        xmin, xmax = np.min(corners.real), np.max(corners.real)
        ymin, ymax = np.min(corners.imag), np.max(corners.imag)
        length, height = xmax - xmin, ymax - ymin
        n_tile = 3
        x = np.array(
            [np.linspace(xmin, xmax, resolution) for i in range(n_tile)]
        ).flatten()
        x_tiled = np.array(
            [
                np.linspace(xmin, xmax, resolution) + i * length
                for i in range(n_tile)
            ]
        ).flatten()
        y_tiled = np.array(
            [
                np.linspace(ymin, ymax, resolution) + i * length
                for i in range(n_tile)
            ]
        ).flatten()
        y = np.array(
            [np.linspace(ymin, ymax, resolution) for i in range(n_tile)]
        ).flatten()
        self.X, self.Y = np.meshgrid(x, y)
        self.X_tiled, self.Y_tiled = np.meshgrid(x_tiled, y_tiled)
        self.Z = self.Y + 1j * self.X
        psi, uv, p, omega = self.solver.functions
        self.psi_values = psi(self.Z.flatten()).reshape(
            n_tile * resolution, n_tile * resolution
        )
        # need to add a for every x and y for every b
        zeros = np.zeros((resolution, resolution))
        a, b = 2, -6
        psi_correction = np.block(
            [
                [zeros, a + zeros, 2 * a + zeros],
                [b + zeros, b + a + zeros, b + 2 * a + zeros],
                [2 * b + zeros, 2 * b + a + zeros, 2 * b + 2 * a + zeros],
            ]
        )
        self.psi_values += psi_correction
        self.uv_values = uv(self.Z.flatten()).reshape(
            n_tile * resolution, n_tile * resolution
        )
        fig, ax = plt.subplots()
        speed = np.abs(self.uv_values)
        parula.set_bad("white")
        pc = ax.pcolormesh(
            self.X_tiled, self.Y_tiled, speed, cmap=parula, shading="gouraud"
        )
        plt.colorbar(pc)
        ax.contour(
            self.X_tiled,
            self.Y_tiled,
            self.psi_values,
            colors="k",
            levels=100,
            linestyles="solid",
            linewidths=0.5,
        )
        ax.set_aspect("equal")
        return fig, ax
