"""Create a polygonal domain using the Domain class.

Raises
------
ValueError: Domain has less than 3 corners
TypeError: Corners must be a list of complex numbers
TypeError: Number of boundary points must be a non negative integer
TypeError: Number of poles must be a non negative integer
TypeError: Spacing must be clustered or linear
TypeError: Point must be a complex number
"""
import numpy as np  # noqa: D100
import matplotlib.pyplot as plt
from numbers import Number
from pyls.numerics import cart, cluster
import shapely
from shapely import Point, Polygon


class Domain:
    """Create a polygonal domain from a list of corners.

    Attributes
    ----------
    corners : (K, 1) array_like
        The corners of the domain.
    num_boundary_points : int
        The number of boundary points to use on each edge.
    num_poles : int
        The number of poles to use in each pole group.
    sigma : float
        The clustering parameter.
    L : float
        The characteristic length of the domain.
    boundary_points : (M, 1) array_like
        The boundary points of the domain.
    sides : list of strings
        The side labels of the domain.
    indices : dictionary of lists of ints
        The indices of the boundary points for each side label.
    poles : list of lists of complex numbers
        The poles of the rational basis.
    polygon : shapely.geometry.Polygon
        The polygon representing the domain.

    Methods
    -------
    check_input():
        Check that the input is valid.
    generate_boundary_points():
        Generate the boundary points.
    generate_poles():
        Generate the poles.
    show():
        Show the labelled domain.
    name_side():
        Name the sides of the domain.

    Notes
    -----
    The corners must be in anticlockwise order.
    """

    def __init__(
        self,
        corners,
        num_boundary_points=100,
        num_poles=24,
        sigma=4,
        L=1,
        spacing="clustered",
    ):
        self.corners = np.array(corners)
        self.num_boundary_points = num_boundary_points
        self.num_poles = num_poles
        self.sigma = sigma
        self.L = L
        self.spacing = spacing
        self.check_input()
        self.polygon = Polygon(
            np.array([self.corners.real, self.corners.imag]).T
        )
        self.orient()
        self.generate_boundary_points()
        self.generate_poles()

    def check_input(self):
        """Check that the input is valid."""
        if len(self.corners) <= 2:
            raise ValueError("Domain must have at least 3 corners")
        for corner in self.corners:
            if not isinstance(corner, Number):
                raise TypeError("Corners must be a list of complex numbers")
        if (
            type(self.num_boundary_points) != int
            or self.num_boundary_points <= 0
        ):
            raise TypeError("num_boundary_points must be a positive integer")
        if type(self.num_poles) != int or self.num_poles < 0:
            raise TypeError("num_poles must be a non negative integer")
        if self.spacing not in ["clustered", "linear"]:
            raise ValueError("spacing must be clustered or linear")

    def orient(self):
        """Reorient the polygon so that it is anticlockwise."""
        self.polygon = shapely.geometry.polygon.orient(self.polygon, 1)
        corners = np.array(self.polygon.exterior.coords[:-1])
        self.corners = corners[:, 0] + 1j * corners[:, 1]

    def generate_boundary_points(self):
        """Create a list of boundary points on each edge."""
        if self.spacing == "clustered":
            spacing = (
                np.tanh(np.linspace(-10, 10, self.num_boundary_points)) + 1
            ) / 2
        elif self.spacing == "linear":
            spacing = np.linspace(0, 1, self.num_boundary_points)
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
                    j * self.num_boundary_points,
                    (j + 1) * self.num_boundary_points,
                )
            ]
            for j, side in enumerate(self.sides)
        }

    def name_side(self, old, new):
        """Rename the sides of the polygon."""
        old, new = str(old), str(new)
        self.sides[self.sides.index(old)] = new
        self.indices[new] = self.indices.pop(old)

    def group_sides(self, old_sides, new):
        """Rename a list of side labels as a single side label."""
        old_sides = [str(side) for side in old_sides]
        self.indices[str(new)] = []
        for side in old_sides:
            self.sides.remove(side)
            self.indices[str(new)] += self.indices.pop(side)
        self.sides.append(str(new))

    def generate_poles(self):
        """Generate exponentially clustered lighting poles."""
        pole_spacing = cluster(self.num_poles, self.L, self.sigma)
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

    def show(self):
        """Display the labelled polygon."""
        fig, ax = plt.subplots()
        flat_poles = self.poles.flatten()
        if len(self.poles[0]):
            x_min = min(flat_poles.real)
            x_max = max(flat_poles.real)
            y_min = min(flat_poles.imag)
            y_max = max(flat_poles.imag)
        else:
            x_min = min(self.corners.real)
            x_max = max(self.corners.real)
            y_min = min(self.corners.imag)
            y_max = max(self.corners.imag)
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)
        cartesian_corners = np.array([self.corners.real, self.corners.imag]).T
        polygon = plt.Polygon(cartesian_corners, fill=True, alpha=0.5)
        ax.add_patch(polygon)
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
        total_points = self.num_boundary_points * nc
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

    def mask_contains(self, z):
        """Return a mask of points in z that are in the polygon."""
        mask = np.zeros(z.shape, dtype=bool)
        it = np.nditer(z, flags=["multi_index"])
        for point in it:
            mask[it.multi_index] = self.__contains__(point)
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
