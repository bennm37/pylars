"""Create a polygonal domain from a list of corners.

Raises:
    ValueError: Domain has less than 3 corners
    TypeError: Corners must be a list of complex numbers
    TypeError: Number of boundary points must be a non negative integer
"""
import numpy as np  # noqa: D100
import matplotlib.pyplot as plt
from numbers import Number, Integral
from collections.abc import Sequence
from collections import Counter
from pylars.numerics import cart, cluster
from shapely import Point, Polygon, LineString
import matplotlib.patches as patches


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
        self._generate_lightning_poles()
        self.polygon = Polygon(
            np.array([self.corners.real, self.corners.imag]).T
        )
        self.interior_curves = []
        self.centroids = {}
        self.movers = []
        self.interior_laurents = []
        self.exterior_laurents = []
        self.interior_laurent_indices = {}
        self.exterior_laurent_indices = {}
        self.mirror_indices = {}

    def check_input(self):
        """Check that the input is valid."""
        if len(self.corners) <= 2:
            raise ValueError("Domain must have at least 3 corners")
        for corner in self.corners:
            if not isinstance(corner, Number):
                raise TypeError("Corners must be a list of complex numbers")
        if (
            not isinstance(self.num_edge_points, Integral)
        ) or self.num_edge_points <= 0:
            raise TypeError("num_edge_points must be a positive integer")
        if (not isinstance(self.num_poles, Integral)) or self.num_poles < 0:
            raise TypeError("num_poles must be a non negative integer")
        if self.spacing != "linear" and self.spacing != "clustered":
            raise ValueError("spacing must be 'linear' or 'clustered'")

    def add_interior_curve(
        self,
        f,
        centroid,
        num_points=100,
        deg_laurent=10,
        aaa=False,
        mirror_laurents=False,
    ):
        """Create an interior curve from a parametric function."""
        side = self._generate_interior_curve_points(f, num_points)
        self._generate_interior_laurent_series(side, deg_laurent, centroid)
        if mirror_laurents:
            self._generate_mirror_laurents(side, deg_laurent, centroid)
        self._update_polygon()

    def add_mover(
        self,
        f,
        centroid,
        num_points=100,
        deg_laurent=10,
        aaa=False,
        mirror_laurents=False,
    ):
        """Create an interior curve from a parametric function."""
        side = self._generate_interior_curve_points(f, num_points)
        self._generate_interior_laurent_series(side, deg_laurent, centroid)
        if mirror_laurents:
            self._generate_mirror_laurents(side, deg_laurent, centroid)
        self._update_polygon()
        self.movers += [side]
        return side

    def remove(self, indices):
        """Remove the points at the given indices and adjust the indices."""
        n = len(self.boundary_points)
        if isinstance(indices, Sequence) or isinstance(indices, np.ndarray):
            indices = np.array(indices).astype(int)
        elif isinstance(indices, Integral):
            indices = np.array([indices]).astype(int)
        else:
            raise TypeError("indices must be a sequence of integers")
        if np.any(indices < 0):
            raise ValueError("indices must be non negative")
        if np.any(indices >= n):
            raise ValueError("indices must be less than the number of points")
        self.boundary_points = np.delete(self.boundary_points, indices, axis=0)
        indices_start_values = [val[0] for val in self.indices.values()]
        # sort the sides so they are in increasing order
        self.sides = np.array(self.sides)[np.argsort(indices_start_values)]
        indices_start_values = np.sort(indices_start_values)
        indices_end_values = np.append(indices_start_values[1:], n)
        index_bins = (
            np.digitize(indices, indices_start_values) - 1
        )  # so that 0 is the first bin
        d_dict = Counter(index_bins)
        decreases = np.zeros(len(self.sides))
        for j, decrease in zip(list(d_dict.keys()), list(d_dict.values())):
            decreases[j] = decrease
        cum_decrease = [0] + np.cumsum(decreases).tolist()
        for i, side in enumerate(self.sides):
            self.indices[side] = np.arange(
                indices_start_values[i] - cum_decrease[i],
                indices_end_values[i] - cum_decrease[i + 1],
                dtype=int,
            )

    def translate(self, side, disp):
        """Translate a side of the domain."""
        if side not in self.sides:
            raise ValueError(f"Side {side} does not exist")
        if not isinstance(disp, Number):
            raise TypeError("disp must be a complex number")
        self.boundary_points[self.indices[side]] += disp
        self.centroids[side] += disp
        for index in self.interior_laurent_indices[side]:
            old_centroid, degree = self.interior_laurents[index]
            new_laurent = (old_centroid + disp, degree)
            self.interior_laurents[index] = new_laurent
        # TODO update mirror laurents
        self._update_polygon()

    def rotate(self, side, angle):
        """Rotate a side of the domain."""
        if side not in self.sides:
            raise ValueError(f"Side {side} does not exist")
        if not isinstance(angle, Number):
            raise TypeError("angle must be a complex number")
        points = self.boundary_points[self.indices[side]]
        centroid = self.centroids[side]
        new_points = centroid + (points - centroid) * np.exp(1j * angle)
        self.boundary_points[self.indices[side]] = new_points
        self._update_polygon()

    def area(self, side):
        """Calculate the area enclosed by an interior curve."""
        if side not in self.interior_curves:
            raise ValueError("Area can only be evaluated on interior curves.")
        points = self.boundary_points[self.indices[side]].reshape(-1)
        poly = Polygon(np.array([points.real, points.imag]).T)
        return poly.area

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
        side = str(len(self.sides))
        self.sides += [side]
        self.interior_curves += [side]
        n_bp = len(self.boundary_points)
        self.indices[side] = [i for i in range(n_bp, n_bp + num_points)]
        self.boundary_points = np.concatenate(
            [self.boundary_points, points.reshape(-1, 1)], axis=0
        )
        return side

    def _name_side(self, old, new):
        """Rename the sides of the polygon."""
        if isinstance(old, str) and isinstance(new, str):
            if old not in self.sides:
                raise ValueError(f"Side {old} does not exist")
            if new in self.sides:
                raise ValueError(f"Side {new} already exists")
            self.sides[self.sides.index(old)] = new
            self.indices[new] = self.indices.pop(old)
        else:
            raise TypeError("Side names must be strings")

    def _group_sides(self, old_sides, new):
        """Rename a list of side labels as a single side label."""
        self.indices[str(new)] = []
        for side in old_sides:
            self.sides.remove(side)
            self.indices[str(new)] += self.indices.pop(side)
        self.sides.append(str(new))

    def _generate_lightning_poles(self):
        """Generate exponentially clustered lightning poles.

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

    def _generate_interior_laurent_series(self, side, degree, centroid):
        """Generate Laurent series in a hole of the domain."""
        interior_points = self.boundary_points[self.indices[side]]
        if centroid is None:
            centroid = np.mean(interior_points)
        self.centroids[side] = centroid
        self.interior_laurents.append((centroid, degree))
        if side not in self.interior_laurent_indices.keys():
            self.interior_laurent_indices[side] = [
                len(self.interior_laurents) - 1
            ]
        else:
            self.interior_laurent_indices[side] += [
                len(self.interior_laurents) - 1
            ]

    def _generate_exterior_laurent_series(self, side, degree, centroid):
        """Generate Laurent series outside the domain."""
        self.exterior_laurents.append((centroid, degree))
        if side not in self.exterior_laurent_indices.keys():
            self.exterior_laurent_indices[side] = [
                len(self.exterior_laurents) - 1
            ]
        else:
            self.exterior_laurent_indices[side] += [
                len(self.exterior_laurents) - 1
            ]

    def _generate_mirror_laurents(self, side, degree, centroid, tol=2):
        """Generate mirror images of the Laurent series."""
        # for each side of the polygon, calculate the inwards
        # facing unit normal vector
        n_corners = len(self.corners)
        corners = np.array(self.corners)
        normals = np.zeros((n_corners), dtype=np.complex128)
        tangents = self.corners - np.roll(self.corners, 1)
        normals = 1j * tangents
        normals = normals / np.abs(normals)
        x2, y2 = corners.real, corners.imag
        x1, y1 = np.roll(corners, 1).real, np.roll(corners, 1).imag
        x0, y0 = centroid.real, centroid.imag
        distances = np.abs(
            (x2 - x1) * (y1 - y0) + (x1 - x0) * (y2 - y1)
        ) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        mirrors = centroid - 2 * normals * distances
        for mirror in mirrors:
            if np.abs(mirror - centroid) < tol:
                self._generate_exterior_laurent_series(side, degree, mirror)
                if side not in self.mirror_indices.keys():
                    self.mirror_indices[side] = [
                        len(self.exterior_laurents) - 1
                    ]
                else:
                    self.mirror_indices[side] += [
                        len(self.exterior_laurents) - 1
                    ]

    def _update_polygon(self, buffer=0):
        """Update the polygon."""
        holes = [
            np.array(
                [
                    self.boundary_points[self.indices[curve]].reshape(-1).real,
                    self.boundary_points[self.indices[curve]].reshape(-1).imag,
                ]
            ).T
            for curve in self.interior_curves
        ]
        self.polygon = Polygon(
            np.array([self.corners.real, self.corners.imag]).T, holes=holes
        )
        if buffer > 0:
            self.polygon = self.polygon.buffer(buffer)

    def enlarge_holes(self, scale_factor):
        """Enlarge the holes in the polygon."""
        for interior_curve in self.interior_curves:
            points = self.boundary_points[self.indices[interior_curve]]
            centroid = self.centroids[interior_curve]
            enlarged_points = (points - centroid) * scale_factor + centroid
            self.boundary_points[self.indices[interior_curve]] = (
                enlarged_points
            ).reshape(-1, 1)
        self._update_polygon()

    def plot(self, figax=None, set_lims=True):
        """Display the labelled polygon."""
        if figax is None:
            fig, ax = plt.subplots()
        flat_poles = self.poles.flatten()
        if set_lims:
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
        for side in self.sides:
            print(side)
            centroid = np.mean(self.boundary_points[self.indices[side]])
            x, y = centroid.real, centroid.imag
            print("using centroid")
            ax.text(
                x,
                y,
                str(side),
                fontsize=20,
                ha="center",
                va="center",
            )

        total_points = len(self.boundary_points)
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

        lightning_poles = ax.scatter(
            flat_poles.real,
            flat_poles.imag,
            c="red",
            s=10,
        )
        handles = [lightning_poles]
        degrees = []
        degree_labels = []
        for centroid, degree in self.interior_laurents:
            laruent = ax.scatter(
                centroid.real,
                centroid.imag,
                c=[(0, (1 - np.exp(-degree / 10)), 0)],
                s=10,
            )
            if degree not in degrees:
                degrees.append(degree)
                handles.append(laruent)
                degree_labels.append(f"Laurent series ({degree}) with log")
        for centroid, degree in self.exterior_laurents:
            laruent = ax.scatter(
                centroid.real,
                centroid.imag,
                c=[(0, (1 - np.exp(-degree / 10)), 0)],
                s=10,
            )
            if degree not in degrees:
                degrees.append(degree)
                handles.append(laruent)
                degree_labels.append(f"Laurent series ({degree})")
        ax.legend(
            handles=handles,
            labels=["Lightning poles"] + degree_labels,
            loc="upper center",
        )
        ax.set_aspect("equal")
        # plt.tight_layout()
        return fig, ax

    def plot_polygon(self, ax, poly):
        """Plot a polygon on the given axis."""
        exterior_coords = np.array(self.polygon.exterior.coords)
        exterior_patch = patches.Polygon(exterior_coords)
        ax.add_patch(exterior_patch)
        for inner in poly.interiors:
            interior_patch = patches.Polygon(
                np.array(inner.coords), color="white"
            )
            ax.add_patch(interior_patch)
        ax.autoscale_view()

    def simple_poles_in_polygon(self, polygon):
        """Return the indices of simple poles inside a polygon."""
        if isinstance(polygon, Sequence):
            polygon = np.array(polygon)
            poly = Polygon(np.array([polygon.real, polygon.imag]).T)
            indices = []
            i = self.deg_poly
            for pole_group in self.poles:
                for pole in pole_group:
                    if poly.contains(Point([pole.real, pole.imag])):
                        indices.append(i)
                    i += 1
            for laurents in self.interior_laurents:
                centroid, degree = laurents
                if poly.contains(Point([centroid.real, centroid.imag])):
                    indices.append(i)
                i += degree
            return indices
        else:
            raise TypeError("polygon must be a sequence of complex numbers")

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
