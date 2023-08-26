"""Analysis module for the pylars package.

Supports plotting of the contours and velocity magnitude of the solution.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pylars.colormaps import parula
from numbers import Integral
import matplotlib.patches as patches
from scipy.integrate import quad
import re


class Analysis:
    """Class for analyzing the solution of a lightning stokes problem.

    Attributes
    ----------
    solution : Solution

    Methods
    -------
    plot():
        Plot the contours and velocity magnitude of the solution.
    """

    def __init__(self, solution):
        self.solution = solution
        self.problem = solution.problem
        self.domain = solution.problem.domain

    def plot(
        self,
        resolution=100,
        n_streamlines=20,
        streamline_type="linear",
        interior_patch=False,
        enlarge_patch=1.0,
        epsilon=1e-3,
        figax=None,
        colorbar=True,
        vmin=None,
        vmax=None,
        quiver=False,
        imshow=False,
    ):
        """Plot the contours and velocity magnitude of the solution."""
        dom = self.domain
        exterior_points = dom.exterior_points
        xmin, xmax = np.min(exterior_points.real), np.max(exterior_points.real)
        ymin, ymax = np.min(exterior_points.imag), np.max(exterior_points.imag)
        x = np.linspace(xmin + epsilon, xmax - epsilon, resolution)
        y = np.linspace(ymin + epsilon, ymax - epsilon, resolution)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.Z = self.X + 1j * self.Y
        psi, uv, p, omega, eij = self.solution.functions
        self.Z[~dom.mask_contains(self.Z)] = np.nan
        self.psi_values = psi(self.Z.flatten()).reshape(resolution, resolution)
        self.uv_values = uv(self.Z.flatten()).reshape(resolution, resolution)
        if figax is not None:
            fig, ax = figax
        else:
            fig, ax = plt.subplots()
        speed = np.abs(self.uv_values)
        parula.set_bad("white")
        if imshow:
            pc = ax.imshow(
                speed.T,
                cmap=parula,
                vmin=vmin,
                vmax=vmax,
                interpolation="quadric",
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
            )
        else:
            pc = ax.pcolormesh(
                self.X,
                self.Y,
                speed,
                cmap=parula,
                shading="gouraud",
                vmin=vmin,
                vmax=vmax,
            )

        if colorbar:
            cb = plt.colorbar(pc)
        if not isinstance(n_streamlines, Integral):
            raise TypeError("n_streamlines must be an integer")
        if streamline_type == "starting_points":
            stride = int(resolution / n_streamlines)
            levels = self.psi_values[0, ::stride]
            levels.sort()
        elif streamline_type == "linear":
            levels = n_streamlines
        else:
            raise ValueError(
                "streamline_type must be 'starting_points' or 'linear'"
            )
        ax.contour(
            self.X,
            self.Y,
            self.psi_values,
            colors="k",
            levels=levels,
            linestyles="solid",
            linewidths=0.5,
        )
        if interior_patch:
            if dom.interior_curves is not None:
                for interior_curve in dom.interior_curves:
                    points = dom.boundary_points[dom.indices[interior_curve]]
                    centroid = dom.centroids[interior_curve]
                    enlarged_points = (
                        points - centroid
                    ) * enlarge_patch + centroid
                    points = np.array(
                        [enlarged_points.real, enlarged_points.imag]
                    ).T.reshape(-1, 2)
                    poly = patches.Polygon(points, color="w", zorder=2)
                    ax.add_patch(poly)

        if quiver:
            stride = 20
            ax.quiver(
                self.X[::stride, ::stride],
                self.Y[::stride, ::stride],
                self.uv_values.real[::stride, ::stride],
                self.uv_values.imag[::stride, ::stride],
                color="gray",
                scale=10,
                zorder=1,
            )
        ax.set_aspect("equal")
        if self.solution.status == "d":
            ax.set(xlabel="x (m)", ylabel="y (m)")
            cb.set_label("Velocity magnitude (m/s)")
        return fig, ax

    def get_Z(self, resolution=100, epsilon=1e-3):  # noqa: N802
        """Get the Z array for plotting contours and velocity magnitude."""
        corners = self.domain.corners
        xmin, xmax = np.min(corners.real), np.max(corners.real)
        ymin, ymax = np.min(corners.imag), np.max(corners.imag)
        x = np.linspace(xmin + epsilon, xmax - epsilon, resolution)
        y = np.linspace(ymin + epsilon, ymax - epsilon, resolution)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.Z = self.X + 1j * self.Y
        return self.X, self.Y, self.Z

    def plot_periodic(
        self,
        resolution=100,
        interior_patch=True,
        quiver=False,
        n_streamlines=50,
        vmax=None,
        n_tile=3,
        figax=None,
        colorbar=True,
        enlarge_patch=1.0,
    ):
        """Plot the contours and velocity magnitude of the solution."""
        corners = self.domain.corners
        xmin, xmax = np.min(corners.real), np.max(corners.real)
        ymin, ymax = np.min(corners.imag), np.max(corners.imag)
        length, height = xmax - xmin, ymax - ymin
        self.X, self.Y, self.Z = self.get_Z(resolution=resolution)
        self.Z[~self.domain.mask_contains(self.Z)] = np.nan
        x_tiled = np.array(
            [
                np.linspace(xmin, xmax, resolution) + i * length
                for i in range(-n_tile // 2 + 1, n_tile // 2 + 1)
            ]
        ).flatten()
        y_tiled = np.array(
            [
                np.linspace(ymin, ymax, resolution) + i * length
                for i in range(-n_tile // 2 + 1, n_tile // 2 + 1)
            ]
        ).flatten()
        self.X_tiled, self.Y_tiled = np.meshgrid(
            x_tiled, y_tiled, indexing="ij"
        )

        psi, uv, p, omega, eij = self.solution.functions
        self.psi_values = psi(self.Z).reshape(resolution, resolution)
        self.psi_values_tiled = np.tile(self.psi_values, (n_tile, n_tile))
        dlr = np.mean(self.psi_values[-1, :] - self.psi_values[0, :])
        dtb = np.mean(self.psi_values[:, -1] - self.psi_values[:, 0])
        ones = np.ones((resolution, resolution))
        psi_correction = np.block(
            [
                [-dlr * ones - dtb, 0 - dtb * ones, dlr * ones - dtb],
                [-dlr * ones, 0 * dtb * ones, dlr * ones],
                [-dlr * ones + dtb, 0 + dtb * ones, dlr * ones + dtb],
            ]
        ).T
        self.psi_values_tiled += psi_correction
        self.uv_values = uv(self.Z).reshape(resolution, resolution)
        self.uv_values_tiled = np.tile(self.uv_values, (n_tile, n_tile))
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        speed = np.abs(self.uv_values_tiled)
        parula.set_bad("white")
        pc = ax.pcolormesh(
            self.X_tiled,
            self.Y_tiled,
            speed,
            vmax=vmax,
            cmap=parula,
            shading="gouraud",
        )
        if colorbar:
            cb = plt.colorbar(pc)
        ax.contour(
            self.X_tiled,
            self.Y_tiled,
            self.psi_values_tiled,
            colors="k",
            levels=n_streamlines,
            linestyles="solid",
            linewidths=0.5,
        )
        disps = np.array(
            [
                [-length, height],
                [0, height],
                [length, height],
                [-length, 0],
                [0, 0],
                [length, 0],
                [-length, -height],
                [0, -height],
                [length, -height],
            ]
        )
        dom = self.domain
        if interior_patch:
            if self.domain.interior_curves is not None:
                for interior_curve in self.domain.interior_curves:
                    points = dom.boundary_points[dom.indices[interior_curve]]
                    centroid = dom.centroids[interior_curve]
                    enlarged_points = (
                        points - centroid
                    ) * enlarge_patch + centroid
                    points = np.array(
                        [enlarged_points.real, enlarged_points.imag]
                    ).T.reshape(-1, 2)
                    for disp in disps:
                        translated_points = points + disp[np.newaxis, :]
                        poly = patches.Polygon(
                            translated_points, color="w", zorder=2
                        )
                        ax.add_patch(poly)
        if quiver:
            stride = 20
            ax.quiver(
                self.X_tiled[::stride, ::stride],
                self.Y_tiled[::stride, ::stride],
                self.uv_values_tiled.real[::stride, ::stride],
                self.uv_values_tiled.imag[::stride, ::stride],
                color="gray",
                scale=10,
                zorder=1,
            )
        # add dashed lines to show the borders
        ax.vlines(
            np.linspace(xmin - length, xmax + length, n_tile + 1),
            ymin - height,
            ymax + height,
            color="k",
            linestyles="dashed",
            linewidths=1,
        )
        ax.hlines(
            np.linspace(ymin - height, ymax + height, n_tile + 1),
            xmin - length,
            xmax + length,
            color="k",
            linestyles="dashed",
            linewidths=1,
        )
        ax.set_aspect("equal")
        # plt.tight_layout()
        return fig, ax

    def plot_relative_periodicity_error(
        self, p_drop_lr=2, p_drop_tb=0, tol=1e-10
    ):
        """Plot the relative periodicity error."""
        errors = self.get_relative_periodicity_errors(tol=tol)
        fig, ax = plt.subplots()
        for name, values in errors.items():
            ax.plot(self.left.imag, values[0], label=f"{name} lr")
            ax.plot(self.top.real, values[1], label=f"{name} tb")
        ax.legend()
        plt.show()

    def plot_stress_torque(self, curve, deriv, centroid):
        """Plot stress and torque."""
        pass

    def bar_relative_periodicity_error(
        self, p_drop_lr=2, p_drop_tb=0, tol=1e-10
    ):
        """Barchart of the max relative periodicity error."""
        errors = self.get_relative_periodicity_errors(tol=tol)
        fig, ax = plt.subplots()
        names = []
        values = []
        for name, value in errors.items():
            names.append(name + " lr")
            values.append(np.max(value[0]))
            names.append(name + " tb")
            values.append(np.max(value[1]))
        indices = np.argsort(values)[::-1]
        values = np.sort(values)[::-1]
        names = np.array(names)[indices]
        cmap = cm.get_cmap("bone")
        # colors = np.log(values) - np.min(np.log(values))
        colors = np.linspace(0, 1, len(values))
        colors /= np.max(colors)
        colors = cmap(colors)
        ax.bar(names, values, color=colors)
        ax.set(yscale="log")
        plt.show()

    def get_wss_data(self, curves, samples):
        return np.array([1])

    def get_permeability(self, curve, curve_deriv, delta_x, delta_p):
        """Calculate the permeability of the domain at the outlet."""
        volume_flux = self.get_volume_flux(curve, curve_deriv)
        curve_length = self.get_length(curve_deriv)
        avg_volume_flux = volume_flux / curve_length
        # calculate the permeability
        k = avg_volume_flux * self.solution.mu * delta_x / (delta_p)
        return k

    def get_volume_flux(self, curve, curve_deriv):
        """Calculate the volume flux over a curve.

        The curve should be parametrised by s in [0, 1].
        """

        def integrand(s):
            dx = np.abs(curve_deriv(s))
            normal = (-curve_deriv(s) * 1j) / dx
            return np.real(np.conj(self.solution.uv(curve(s))) * normal) * dx

        return quad(integrand, 0, 1)[0]

    def get_length(self, curve_deriv):
        """Calculate the length of a curve."""
        return quad(lambda s: np.abs(curve_deriv(s)), 0, 1)[0]

    def get_relative_periodicity_errors(self, tol=1e-10):
        """Calculate the relative periodicity errors for all variables."""
        e11_error_lr, e11_error_tb = self.get_relative_periodicity_error(
            lambda x: self.solution.eij(x)[:, 0, 0], tol=tol
        )
        e12_error_lr, e12_error_tb = self.get_relative_periodicity_error(
            lambda x: self.solution.eij(x)[:, 0, 1], tol=tol
        )
        p_error_lr, p_error_tb = self.get_relative_periodicity_error(
            lambda x: self.solution.eij(x)[:, 0, 1], tol=tol
        )
        u_error_lr, u_error_tb = self.get_relative_periodicity_error(
            lambda x: self.solution.eij(x)[:, 0, 1], tol=tol
        )
        v_error_lr, v_error_tb = self.get_relative_periodicity_error(
            lambda x: self.solution.eij(x)[:, 0, 1], tol=tol
        )
        return {
            "e11": [e11_error_lr, e11_error_tb],
            "e12": [e12_error_lr, e12_error_tb],
            "p": [p_error_lr, p_error_tb],
            "u": [u_error_lr, u_error_tb],
            "v": [v_error_lr, v_error_tb],
        }

    def get_relative_periodicity_error(self, f, tol=1e-10):
        """Calculate the relative periodicity error."""
        if not hasattr(self, "left") or not hasattr(self, "top"):
            self.get_left_top()
        left_error = np.abs(f(self.left) - f(self.left + 2)) / np.max(
            np.abs(f(self.left + 2))
        )
        top_error = np.abs(f(self.top) - f(self.top - 2j)) / np.max(
            np.abs(f(self.top - 2j))
        )
        if np.max(np.abs(f(self.top - 2j))) < tol:
            print("Warning: top error is rounded to zero")
            top_error = np.zeros_like(top_error)
        if np.max(np.abs(f(self.left + 2))) < tol:
            print("Warning: left error is rounded to zero")
            top_error = np.zeros_like(top_error)
        return left_error, top_error

    def get_left_top(self):
        """Get points on left and bottom sides."""
        self.top = self.domain.boundary_points[self.domain.indices["0"]]
        self.left = self.domain.boundary_points[self.domain.indices["1"]]

    def non_dimensionalise(self, sol):
        """Non-dimensionalise the solution."""
        Z = self.get_Z(resolution=200)
        U = np.max(np.abs(sol.uv(Z)))
        # non-dimensionalise
        nd_sol = self.sol / U
        nd_sol.problem.scale_boundary_conditions(U)
        return nd_sol

    def save_pgf(self, filename, image_root=None):
        """Save the solution to a pgf file."""
        import matplotlib

        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
        plt.savefig(filename + ".pgf", backend="pgf")
        # if image_root is not None:
        #     raise NotImplementedError
        # if image_root is None:
        #     image_root = "figures/python_figures/pgf_files"
