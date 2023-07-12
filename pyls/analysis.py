"""Analysis module for the pyls package.

Supports plotting of the contours and velocity magnitude of the solution.
"""
import numpy as np
import matplotlib.pyplot as plt
from pyls.colormaps import parula
from matplotlib.animation import FuncAnimation


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
        x = np.linspace(xmin, xmax, resolution)
        y = np.linspace(ymin, ymax, resolution)
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

    def plot_periodic(
        self,
        a,
        b,
        gapa=2,
        gapb=2,
        resolution=100,
        n_tile=3,
        figax=None,
        colorbar=True,
    ):
        """Plot the contours and velocity magnitude of the solution."""
        corners = self.domain.corners
        xmin, xmax = np.min(corners.real), np.max(corners.real)
        ymin, ymax = np.min(corners.imag), np.max(corners.imag)
        length, height = xmax - xmin, ymax - ymin
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
        self.X_tiled, self.Y_tiled = np.meshgrid(
            x_tiled, y_tiled, indexing="ij"
        )
        self.Z = self.X + 1j * self.Y
        psi, uv, p, omega = self.solver.functions
        if gapa is None:
            gapa = np.around(psi(ymax) - psi(ymin), 15)
        if gapb is None:
            gapb = np.around(psi(xmax) - psi(xmin), 15)
        self.psi_values = psi(self.Z).reshape(
            n_tile * resolution, n_tile * resolution
        )
        # need to add a for every x and y for every b
        psi_correction = np.zeros((n_tile * resolution, n_tile * resolution))
        for i in range(n_tile):
            for j in range(n_tile):
                psi_correction[
                    resolution * i : resolution * (i + 1),
                    resolution * j : resolution * (j + 1),
                ] = (
                    gapb * b * i + gapa * a * j
                )
        self.psi_values += psi_correction
        self.uv_values = uv(self.Z).reshape(
            n_tile * resolution, n_tile * resolution
        )
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        speed = np.abs(self.uv_values)
        parula.set_bad("white")
        pc = ax.pcolormesh(
            self.X_tiled, self.Y_tiled, speed, cmap=parula, shading="gouraud"
        )
        if colorbar:
            plt.colorbar(pc)
        ax.contour(
            self.X_tiled,
            self.Y_tiled,
            self.psi_values,
            colors="k",
            levels=30,
            linestyles="solid",
            linewidths=0.5,
        )
        # add dashed lines to show the borders
        ax.vlines(
            np.linspace(xmin, xmin + n_tile * length, n_tile + 1),
            ymin,
            ymin + n_tile * height,
            color="k",
            linestyles="dashed",
            linewidths=1,
        )
        ax.hlines(
            np.linspace(ymin, ymin + n_tile * height, n_tile + 1),
            xmin,
            xmin + n_tile * length,
            color="k",
            linestyles="dashed",
            linewidths=1,
        )
        ax.set_aspect("equal")
        # add text to the corners of the figure
        padx = 0.5
        pady = 1.5
        ax.text(
            xmin * padx,
            ymin * pady,
            f"$\psi = {psi(xmin+1j*ymin)[0][0]:.1e}$",
            color="black",
            fontsize=15,
            horizontalalignment="right",
            verticalalignment="top",
        )
        ax.text(
            xmax * padx + (n_tile - 1) * length,
            ymin * pady,
            f"$\psi = {psi(xmax+1j*ymin)[0][0]:.1e}$",
            color="black",
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="top",
        )
        ax.text(
            xmax * padx + (n_tile - 1) * length,
            ymax * pady + (n_tile - 1) * height,
            f"$\psi = {psi(xmax+1j*ymax)[0][0]:.1e}$",
            color="black",
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="bottom",
        )
        ax.text(
            xmin * padx,
            ymax * pady + (n_tile - 1) * height,
            f"$\psi = {psi(xmin+1j*ymax)[0][0]:.1e}$",
            color="black",
            fontsize=15,
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.set(title=f"${a:.2}\psi_A+{b:.2}\psi_B$")
        # plt.tight_layout()
        return fig, ax

    def animate_combination(
        self, sol_2, a_values, b_values, gapa=2, gapb=2, n_tile=3
    ):
        """Animate a linear combination of the solutions."""
        a, b = a_values[0], b_values[0]
        sol_combined = a * self.solver + b * sol_2
        an = Analysis(self.domain, sol_combined)
        fig, ax = an.plot_periodic(
            a=a, b=b, gapa=gapa, gapb=gapb, n_tile=n_tile, colorbar=False
        )

        def update(i):
            if i % 10 == 0:
                print(f"Animating frame {i}")
            ax.clear()
            a, b = a_values[i], b_values[i]
            sol_combined = a * self.solver + b * sol_2
            an = Analysis(self.domain, sol_combined)
            fig1, ax1 = an.plot_periodic(
                a=a,
                b=b,
                gapa=gapa,
                gapb=gapb,
                n_tile=n_tile,
                resolution=30,
                figax=(fig, ax),
                colorbar=False,
            )

        anim = FuncAnimation(fig, update, frames=len(a_values), interval=200)
        return fig, ax, anim

    def plot_error(self):
        """Plot the error along the boundary."""
        A, b, coeff = self.solver.A, self.solver.b, self.solver.coefficients
        err = np.abs(A @ coeff - b)
        fig, ax = plt.subplots()
        ax.plot(err)
        return fig, ax

    def plot_stream_boundary(self):
        """Plot the stream function along the boundary."""
        points = self.domain.boundary_points
        psi, uv, p, omega = self.solver.functions
        fig, ax = plt.subplots()
        ax.plot(psi(points))
        ax.plot(uv(points).real)
        ax.plot(uv(points).imag)
        ax.legend(["psi", "u", "v"])
        return fig, ax
