"""Class for analysis of simulation results."""
from pylars import Analysis
import numpy as np
import matplotlib.animation as animation


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
            self.angular_velocity_data = self.mover_data[
                "angular_velocities"
            ]
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
