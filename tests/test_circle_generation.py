"""Test circle generation methods."""


def test_generate_circles():
    """Test generate_circles."""
    import numpy as np
    from pylars.domain import generate_circles

    n_circles = 10
    radius = 0.1
    centroids = generate_circles(n_circles, radius)
    assert len(centroids) == n_circles
    assert np.all(np.abs(centroids) < np.sqrt(2))
    assert np.all(np.abs(centroids.imag) < 1)


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
    ax.set_aspect("equal")
    # plt.show()


def test_lognormal_circles():
    """Test generate_rv_circles."""
    from scipy.stats import lognorm
    import numpy as np
    from pylars.domain import generate_rv_circles

    porosity = 0.95
    rv = lognorm.rvs
    rv_args = {"scale": 1.20, "loc": 0.217, "s": 1.27}
    length = 100
    centers, radii = generate_rv_circles(
        porosity, rv, rv_args, length, min_dist=0.01
    )
    print("Circles generated.")
    # plot(centers, radii, length)
    assert len(centers) == len(radii)
    assert np.isclose(
        1 - np.sum(np.pi * radii**2) / length**2, porosity, rtol=1
    )


if __name__ == "__main__":
    test_generate_circles()
    test_lognormal_circles()
