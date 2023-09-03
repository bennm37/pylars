"""Class for analysis of simulation results."""
from pylars import Analysis
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
from pylars.colormaps import parula


class SimulationAnalysis:
    """Class for analysis of simulation results."""

    def __init__(self, results):
        self.results = results
        self.solution_data = results["solution_data"]
        self.time_data = results["time_data"]
        if "mover_data" in results.keys():
            self.mover_data = results["mover_data"]
            self.position_data = self.mover_data["positions"]
            self.velocity_data = self.mover_data["velocities"]
            self.angle_data = self.mover_data["angles"]
            self.angular_velocity_data = self.mover_data["angular_velocities"]
        self.name = "SimulationAnalysis"

    def animate(self, resolution=100, interval=100, vmin=0, vmax=1):
        """Animate Solution Data."""
        # TODO make this faster
        sol0 = self.solution_data[0]
        an = Analysis(sol0)
        fig, ax = an.plot()
        t = 0.0
        ax.set(title=f"t = {t:.2f}")
        if self.mover_data is not None:
            ax.quiver(
                self.position_data[0].real,
                self.position_data[0].imag,
                self.velocity_data[0].real,
                self.velocity_data[0].imag,
                color="k",
            )
            ax.quiver(
                self.position_data[0].real,
                self.position_data[0].imag,
                np.cos(self.angle_data[0]),
                np.sin(self.angle_data[0]),
                color="r",
            )

        def update(i):
            ax.clear()
            ax.set(title=f"t = {self.time_data[i]:.2f}")
            if i % 5 == 0:
                print("Animating frame", i)
            an = Analysis(self.solution_data[i])
            an.plot(
                resolution=resolution,
                interior_patch=True,
                figax=(fig, ax),
                colorbar=False,
                vmin=vmin,
                vmax=vmax,
            )
            ax.title.set_text(f"t = {t:.2f}")
            ax.quiver(
                self.position_data[i].real,
                self.position_data[i].imag,
                self.velocity_data[i].real,
                self.velocity_data[i].imag,
                color="k",
                zorder=4,
                scale=10,
            )
            ax.quiver(
                self.position_data[i].real,
                self.position_data[i].imag,
                np.cos(self.angle_data[i]),
                np.sin(self.angle_data[i]),
                color="r",
                zorder=3,
            )

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.time_data), interval=interval
        )
        return fig, ax, anim

    def animate_fast(
        self,
        resolution=100,
        interval=100,
        n_levels=20,
        streamline_type="starting_points",
    ):
        """Animate Solution Data."""
        parula.set_bad("white")
        self.generate_array_data(resolution, n_levels=n_levels)
        fig, ax = plt.subplots()
        self.speed = np.abs(self.uv_data)
        vmin = self.speed[~np.isnan(self.speed)].min()
        vmax = self.speed[~np.isnan(self.speed)].max()
        pc = ax.pcolormesh(
            self.X,
            self.Y,
            self.speed[0, :, :],
            cmap=parula,
            shading="gouraud",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(pc)
        global contours
        if streamline_type == "starting_points":
            levels = self.levels[0]
        else:
            levels = n_levels
        contours = ax.contour(
            self.X,
            self.Y,
            self.psi_data[0, :, :],
            colors="k",
            levels=levels,
            linestyles="solid",
            linewidths=0.5,
        )
        if self.mover_data is not None:
            velocity = ax.quiver(
                self.position_data[0].real,
                self.position_data[0].imag,
                self.velocity_data[0].real,
                self.velocity_data[0].imag,
                color="k",
                zorder=4,
            )
            direction = ax.quiver(
                self.position_data[0].real,
                self.position_data[0].imag,
                np.cos(self.angle_data[0]),
                np.sin(self.angle_data[0]),
                color="r",
                zorder=4,
            )
        domain = self.solution_data[0].problem.domain
        blobs = []
        global movers
        movers = []
        if domain.interior_curves is not None:
            for interior_curve in domain.interior_curves:
                points = domain.boundary_points[
                    domain.indices[interior_curve]
                ].reshape(-1)
                points = np.array([points.real, points.imag]).T
                poly = patches.Polygon(points, color="w", zorder=2)
                patch = ax.add_patch(poly)
                if interior_curve in domain.movers:
                    movers.append(patch)
                else:
                    blobs.append(patch)

        def update(i):
            global contours
            global movers
            if i % 5 == 0:
                print("Animating frame", i)
            for coll in contours.collections:
                coll.remove()
            pc.set_array(self.speed[i, :, :].flatten())
            if streamline_type == "starting_points":
                levels = self.levels[i]
            else:
                levels = n_levels
            contours = ax.contour(
                self.X,
                self.Y,
                self.psi_data[i, :, :],
                colors="k",
                levels=levels,
                # levels=20,
                linestyles="solid",
                linewidths=0.5,
            )
            if self.mover_data is not None:
                p = np.array(
                    [self.position_data[i].real, self.position_data[i].imag]
                ).T
                velocity.set_offsets(p)
                velocity.set_UVC(
                    self.velocity_data[i].real, self.velocity_data[i].imag
                )
                direction.set_offsets(p)
                direction.set_UVC(
                    np.cos(self.angle_data[i]), np.sin(self.angle_data[i])
                )
                for mover in movers:
                    mover.remove()
                movers = []
                domain = self.solution_data[i].problem.domain
                for mover in domain.movers:
                    points = domain.boundary_points[
                        domain.indices[mover]
                    ].reshape(-1)
                    points = np.array([points.real, points.imag]).T
                    poly = patches.Polygon(points, color="w", zorder=2)
                    patch = ax.add_patch(poly)
                    movers.append(patch)

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.time_data), interval=interval
        )
        return fig, ax, anim

    def generate_array_data(self, resolution, n_levels=20):
        """Generate array data for uv and psi."""
        an = Analysis(self.solution_data[0])
        self.X, self.Y, self.Z = an.get_Z(resolution)
        self.psi_data = np.zeros(
            (len(self.solution_data), resolution, resolution),
            dtype=np.float64,
        )
        self.uv_data = np.zeros(
            (len(self.solution_data), resolution, resolution),
            dtype=np.complex128,
        )
        for i, sol in enumerate(self.solution_data):
            Z = self.Z.copy()
            Z[~sol.problem.domain.mask_contains(self.Z)] = np.nan
            self.psi_data[i, :, :] = sol.psi(Z.flatten()).reshape(
                resolution, resolution
            )
            self.uv_data[i, :, :] = sol.uv(Z.flatten()).reshape(
                resolution, resolution
            )
        self.levels = self.psi_data[:, 0, :: resolution // n_levels].real
        self.levels.sort(axis=1)
        print("finished")

    def generate_pathlines(self, frame, starting_positions=100):
        """Generate pathlines."""
        if isinstance(starting_positions, int):
            starting_positions = (
                -1 + np.linspace(-1, 1, starting_positions) * 1j
            )
        times = self.time_data[: frame + 1]
        n_particles = len(starting_positions)
        n_steps = len(times)
        dt = np.diff(self.time_data)[0]
        pathline_data = np.zeros(
            (n_steps + 1, n_particles), dtype=np.complex128
        )
        pathline_velocity_data = np.zeros(
            (n_steps + 1, n_particles), dtype=np.complex128
        )
        pathline_data[0, :] = starting_positions
        for i in range(n_steps):
            velocities = (
                self.solution_data[i].uv(pathline_data[i, :]).reshape(-1)
            )
            pathline_velocity_data[i, :] = velocities
            pathline_data[i + 1, :] = pathline_data[i, :] + velocities * dt
        self.pathline_data = pathline_data
        self.pathline_velocity_data = pathline_velocity_data

    def plot_pathlines(self, frame, n_particles=500):
        """Plot pathlines."""
        # generate random starting positions in the fluid domain
        starting_positions = np.zeros(n_particles, dtype=np.complex128)
        count = 0
        dom0 = self.solution_data[0].problem.domain
        while count < n_particles:
            position = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
            if position in dom0:
                starting_positions[count] = position
                count += 1

        self.generate_pathlines(frame, starting_positions=starting_positions)

        def colorline(
            x,
            y,
            array,
            cmap=parula,
            norm=plt.Normalize(0.0, 1.0),
            linewidth=0.5,
        ):
            # Create a path from the x and y points
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from data points to colors
            lc = mcoll.LineCollection(
                segments,
                array=array,
                cmap=cmap,
                norm=norm,
                linewidth=linewidth,
            )
            return lc

        fig, ax = plt.subplots()
        dom_frame = self.solution_data[frame].problem.domain
        speeds = np.abs(self.pathline_velocity_data)
        for line, speed in zip(self.pathline_data.T, speeds.T):
            lc = colorline(line.real, line.imag, speed)
            ax.add_collection(lc)
        # ax.plot(
        #     self.pathline_data.real,
        #     self.pathline_data.imag,
        #     color=speed,
        #     alpha=0.5,
        # )
        # plot the movers path
        ax.scatter(
            self.pathline_data[-1].real,
            self.pathline_data[-1].imag,
            color=parula(speeds[-2, :]),
            s=1,
            zorder=10,
        )
        dom_frame.plot_polygon(
            ax=ax,
            poly=dom_frame.polygon,
            exterior_color="w",
            interior_color="black",
            # zorder=10,
        )
        ax.plot(
            self.position_data[: frame + 1].real,
            self.position_data[: frame + 1].imag,
            color=[0.82, 0.1, 0.26],
            alpha=1,
            zorder=999,
        )
        ax.set_aspect("equal")
        ax.set(xlim=(-1, 1), ylim=(-1, 1))
        ax.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], color="k", linewidth=2)
        ax.axis("off")
        plt.colorbar(lc)
        return fig, ax
