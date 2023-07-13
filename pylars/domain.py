"""Create a polygonal domain from a list of corners.

Raises:
    ValueError: Domain has less than 3 corners
    TypeError: Corners must be a list of complex numbers
    TypeError: Number of boundary points must be a non negative integer
"""
import numpy as np  # noqa: D100
import matplotlib.pyplot as plt
from numbers import Number
from pylars.numerics import cart, cluster
from shapely import Point, Polygon, LineString
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


class Domain:
    """Create a polygonal domain from a list of corners.

    The corners must be in anticlockwise order.
    """

    def __init__(
        self,
        corners,
        num_edge_points=100,
        num_poles=24,
        sigma=4,
        length_scale=1,
        deg_poly=10,
        spacing="clustered",
    ):
        self.corners = np.array(corners)
        self.num_edge_points = num_edge_points
        self.num_poles = num_poles
        self.sigma = sigma
        self.length_scale = length_scale
        self.deg_poly = deg_poly
        self.spacing = spacing
        self.check_input()
        self._generate_exterior_polygon_points()
        self._generate_lighting_poles()
        self.polygon = Polygon(
            np.array([self.corners.real, self.corners.imag]).T
        )
        self.interior_curves = []
        self.laurents = []

    def check_input(self):
        """Check that the input is valid."""
        if len(self.corners) <= 2:
            raise ValueError("Domain must have at least 3 corners")
        for corner in self.corners:
            if not isinstance(corner, Number):
                raise TypeError("Corners must be a list of complex numbers")
        if type(self.num_edge_points) != int or self.num_edge_points <= 0:
            raise TypeError("num_edge_points must be a positive integer")
        if type(self.num_poles) != int or self.num_poles < 0:
            raise TypeError("num_poles must be a non negative integer")
        if self.spacing != "linear" and self.spacing != "clustered":
            raise ValueError("spacing must be 'linear' or 'clustered'")

    def add_interior_curve(
        self, f, num_points=100, deg_laurent=10, aaa=False, mirror=False
    ):
        points = self._generate_interior_curve_points(f, num_points)
        self._generate_laurent_series(points, deg_laurent)
        self._update_polygon()

    def _generate_exterior_polygon_points(self):
        """Create a list of boundary points on each edge."""
        if self.spacing == "linear":
            spacing = np.linspace(0, 1, self.num_edge_points)
        else:
            spacing = (
                np.tanh(np.linspace(-10, 10, self.num_edge_points)) + 1
            ) / 2
        nc = len(self.corners)
        self.boundary_points = np.array(
            [
                self.corners[i]
                + (self.corners[(i + 1) % nc] - self.corners[i]) * spacing
                for i in range(len(self.corners))
            ]
        ).reshape(-1, 1)
        self.sides = [str(i) for i in range(len(self.corners))]
        self.indices = {
            side: [
                i
                for i in range(
                    j * self.num_edge_points,
                    (j + 1) * self.num_edge_points,
                )
            ]
            for j, side in enumerate(self.sides)
        }

    def _generate_interior_curve_points(self, f, num_points):
        if not np.isclose(f(0), f(1)):
            raise ValueError("Curve must be closed")
        points = f(np.linspace(0, 1, num_points)).astype(np.complex128)
        # create a shapely LineString and check it is simple
        # (i.e. does not intersect itself)
        line = LineString(np.array([points.real, points.imag]).T)
        if not line.is_simple:
            raise ValueError("Curve must not intersect itself")
        self.interior_curves.append(points)
        return points

    def _name_side(self, old, new):
        """Rename the sides of the polygon."""
        self.sides[self.sides.index(old)] = new
        self.indices[new] = self.indices.pop(old)

    def _group_sides(self, old_sides, new):
        """Rename a list of side labels as a single side label."""
        self.indices[str(new)] = []
        for side in old_sides:
            self.sides.remove(side)
            self.indices[str(new)] += self.indices.pop(side)
        self.sides.append(str(new))

    def _generate_lighting_poles(self):
        """Generate exponentially clustered lighting poles.

        Poles are clustered exponentially.
        """
        pole_spacing = cluster(self.num_poles, self.length_scale, self.sigma)
        # find the exterior angle bisector at each corner
        bisectors = np.array(
            [
                (self.corners[i] - self.corners[i - 1])
                / np.abs(self.corners[i] - self.corners[i - 1])
                + (self.corners[i] - self.corners[(i + 1) % len(self.corners)])
                / np.abs(
                    self.corners[i] - self.corners[(i + 1) % len(self.corners)]
                )
                for i in range(len(self.corners))
            ]
        )
        bisectors /= np.abs(bisectors)
        # make an array of length len(self.corners) which is one
        # if the midpoint of the line joining the two neighbouring corners
        # is inside the domain and -1 otherwise
        signs = np.array(
            [
                np.sign(
                    np.cross(
                        cart(self.corners[i] - self.corners[i - 1]),
                        cart(
                            self.corners[(i + 1) % len(self.corners)]
                            - self.corners[i]
                        ),
                    )
                )
                for i in range(len(self.corners))
            ]
        )
        self.poles = np.array(
            [
                self.corners[i] + signs[i] * bisectors[i] * pole_spacing
                for i in range(len(self.corners))
            ]
        )

    def _generate_laurent_series(self, interior_points, degree):
        centroid = np.mean(interior_points)
        self.laurents.append((centroid, degree))

    def _update_polygon(self):
        """Update the polygon."""
        holes = [
            np.array([curve.real, curve.imag]).T
            for curve in self.interior_curves
        ]
        self.polygon = Polygon(
            np.array([self.corners.real, self.corners.imag]).T, holes=holes
        )

    def show(self):
        """Display the labelled polygon."""
        fig, ax = plt.subplots()
        flat_poles = self.poles.flatten()
        try:
            x_min = min(flat_poles.real)
            x_max = max(flat_poles.real)
            y_min = min(flat_poles.imag)
            y_max = max(flat_poles.imag)
        except ValueError:
            x_min = min(self.corners.real)
            x_max = max(self.corners.real)
            y_min = min(self.corners.imag)
            y_max = max(self.corners.imag)
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)
        self.plot_polygon(ax, self.polygon)
        # label each edge of the polygon in order
        nc = len(self.corners)
        for i in range(nc):
            x = (self.corners[(i + 1) % nc].real + self.corners[i].real) / 2
            y = (self.corners[(i + 1) % nc].imag + self.corners[i].imag) / 2
            ax.text(
                x,
                y,
                str(i),
                fontsize=20,
                ha="center",
                va="center",
            )
        total_points = self.num_edge_points * nc
        color = np.arange(total_points)
        print(f"Total number of boundary points: {total_points}")
        print(f"Shape of boundary_points: {self.boundary_points.shape}")
        ax.scatter(
            self.boundary_points.real,
            self.boundary_points.imag,
            c=color,
            vmin=0,
            vmax=total_points,
            s=5,
        )
        ax.scatter(
            flat_poles.real,
            flat_poles.imag,
            c="red",
            s=10,
        )
        ax.set_aspect("equal")
        plt.show()

    def plot_polygon(self, ax, poly, **kwargs):
        exterior_coords = np.array(self.polygon.exterior.coords)
        exterior_patch = patches.Polygon(exterior_coords)
        ax.add_patch(exterior_patch)
        for inner in poly.interiors:
            interior_patch = patches.Polygon(
                np.array(inner.coords), color="white"
            )
            ax.add_patch(interior_patch)
        ax.autoscale_view()

    def mask_contains(self, z):
        """Create a boolean mask of where z is in the domain."""
        mask = np.zeros(z.shape, dtype=bool)
        it = np.nditer(z, flags=["multi_index"])
        for z in it:
            mask[it.multi_index] = self.__contains__(z)
        return mask

    def __contains__(self, point):
        """Check if a point is in the polygon."""
        if isinstance(point, complex):
            point = Point(np.array([point.real, point.imag]))
            return self.polygon.contains(point)
        if isinstance(point, np.ndarray):
            if point.dtype != complex or point.dtype != np.complex128:
                raise TypeError("Point must be a complex number or array")
            for p in np.nditer(point):
                if not self.__contains__(complex(p)):
                    return False
            return True
        else:
            return NotImplemented
