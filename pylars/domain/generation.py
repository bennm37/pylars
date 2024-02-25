"""Domain generation functions."""

import numpy as np


def square(z_0, L):
    """Return a parametric curve for a square."""
    return rectangle(z_0, L, L)


def rectangle(z_0, L, H):
    """Return a parametric curve for a rectangle."""
    corners = [
        z_0 + L / 2 + H / 2 * 1j,
        z_0 - L / 2 + H / 2 * 1j,
        z_0 - L / 2 - H / 2 * 1j,
        z_0 + L / 2 - H / 2 * 1j,
    ]
    return polygon(corners)


def polygon(corners):
    """Return a parametric curve for a polygon."""
    lines = [(start, end) for start, end in zip(corners, corners[1:] + corners[:1])]
    lengths = [np.abs(end - start) for start, end in lines]
    bins = np.cumsum(lengths) / np.sum(lengths)

    def curve(t, lines=lines, bins=bins):
        """Return a point on the polygon."""
        i = np.searchsorted(bins, t)
        start, end = lines[i]
        if i == 0:
            prev = 0
        else:
            prev = bins[i - 1]
        t = (t - prev) / (bins[i] - prev)
        return start + t * (end - start)

    return np.vectorize(curve)


def generate_circles(n_circles, radius):
    """Generate non-overlapping circles."""
    L = 1.8
    centroids = np.array(
        L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
    )
    n_current = 1
    i = 0
    while n_current < n_circles:
        i += 1
        if i % 100000 == 0:
            print(i)
            print(f"{n_current=}")
        centroid = L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
        if np.min(np.abs(centroid - centroids)) > 2 * radius:
            centroids = np.append(centroids, centroid)
            n_current += 1
    return centroids


def generate_normal_circles(n_circles, mean, std, seed):
    """Generate non-overlapping circles."""
    np.random.seed(seed)
    L = 2 - 3 * (mean + 5 * std)
    radii = np.array(np.random.normal(mean, std, 1))
    centroids = np.array(
        L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
    )
    n_current = 1
    radius = np.random.normal(mean, std, 1)
    while n_current < n_circles:
        centroid = L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
        if np.min(np.abs(centroid - centroids) / (radii + radius)) > 1.5:
            centroids = np.append(centroids, centroid)
            radii = np.append(radii, radius)
            n_current += 1
            radius = np.random.normal(mean, std, 1)
    return centroids, radii


def generate_normal_circles(n_circles, mean, std):
    """Generate non-overlapping circles."""
    L = 2 - 3 * (mean + 5 * std)
    radii = np.array(np.random.normal(mean, std, 1))
    centroids = np.array(
        L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
    )
    n_current = 1
    radius = np.random.normal(mean, std, 1)
    while n_current < n_circles:
        centroid = L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
        if np.min(np.abs(centroid - centroids) / (radii + radius)) > 1.5:
            centroids = np.append(centroids, centroid)
            radii = np.append(radii, radius)
            n_current += 1
            radius = np.random.normal(mean, std, 1)
    return centroids, radii


def generate_rv_circles(porosity, rv, rv_args, length=2, min_dist=0.05):
    """Generate non-overlapping circles with radii distributed with rv.

    Circles are all at least min_dist away from each other and the walls.
    Terminates when over the target porosity.
    """
    area = length**2
    centers, radii = np.array([]), np.array([])
    too_big_count = 0
    while np.sum(np.pi * radii**2) < (1 - porosity) * area:
        radius = rv(**rv_args)
        bound = length / 2 - radius - min_dist
        if bound < 0:
            too_big_count += 1
            print(
                f"Rejected {too_big_count} circles as radii too large. Make the domain bigger."
            )
            if too_big_count > 1:
                raise ValueError("Radius too big for domain.")
            continue
        center = np.random.uniform(-bound, bound) + 1j * np.random.uniform(
            -bound, bound
        )
        if np.all(np.abs(center - centers) > radius + radii + min_dist):
            centers = np.append(centers, center)
            radii = np.append(radii, radius)
    return centers, radii
