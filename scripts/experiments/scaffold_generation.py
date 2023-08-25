from pylars import Problem, Solver, Analysis
from pylars.domain.generation import generate_rv_circles
from scipy.stats import gamma, lognorm
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def plot(centers, radii, length):
    """Plot circles."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np

    fig, ax = plt.subplots()
    for center, radius in zip(centers, radii):
        c = np.array([center.real, center.imag])
        ax.add_patch(Ellipse(c, 2 * radius, 2 * radius, fill=True, alpha=1.0))
    ax.set_xlim(-length / 2, length / 2)
    ax.set_ylim(-length / 2, length / 2)
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    return fig, ax


if __name__ == "__main__":
    porosity = 0.95
    L = 1e-6
    lengths = np.linspace(10, 50, 10)
    seeds = range(1, 20)
    nc_data = np.zeros((len(lengths), len(seeds)))
    porosity_data = np.zeros((len(lengths), len(seeds)))
    dist = "lognorm"
    for i, length in enumerate(lengths):
        foldername = f"{length*L:.1e}"
        print(f"Starting length {foldername}")
        bound = length / 2
        try:
            os.mkdir(f"media/lognorm_circles/{foldername}")
        except FileExistsError:
            shutil.rmtree(f"media/lognorm_circles/{foldername}")
            os.mkdir(f"media/lognorm_circles/{foldername}")
        for j, seed in enumerate(seeds):
            if dist == "lognorm":
                rv = lognorm.rvs
                rv_args = {"s": 0.5, "scale": 0.275, "loc": 0.0}
            if dist == "gamma":
                rv = lognorm.rvs
                rv_args = {"a": 5, "scale": 0.05, "loc": 0.0}
            np.random.seed(seed)
            centroids, radii = generate_rv_circles(
                porosity=porosity,
                rv=rv,
                rv_args=rv_args,
                length=length,
                min_dist=0.05,
            )
            n_circles = len(centroids)
            # print(f"Number of circles: {n_circles}")
            # print(f"Porosity: {1 - np.sum(np.pi * radii**2) / length**2}")
            fig, ax = plot(centroids * L, radii * L, length * L)
            plt.savefig(
                f"media/lognorm_circles/{foldername}/{dist}_{seed}.pdf",
                bbox_inches="tight",
            )
            plt.close()
            nc_data[i, j] = n_circles
            porosity_data[i, j] = 1 - np.sum(np.pi * radii**2) / length**2
    plt.style.use("ggplot")
    fig, ax = plt.subplots(2, 1)
    err_nc = np.std(nc_data, axis=1)
    ax[0].errorbar(lengths * L, np.mean(nc_data, axis=1), yerr=err_nc)
    ax[0].set_ylabel("Number of circles")
    ax[0].set_xlabel("Length (m)")
    err_porosity = np.std(porosity_data, axis=1)
    ax[1].errorbar(
        lengths * L, np.mean(porosity_data, axis=1), yerr=err_porosity
    )
    ax[1].set_ylabel("Porosity")
    ax[1].set_xlabel("Length (m)")
    plt.show()
