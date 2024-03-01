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
from pylars.numerics import cart, cluster, aaa
from shapely import Point, Polygon, LineString
import matplotlib.patches as patches

BLUE = [0.36, 0.54, 0.66]


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
        self.exterior_points = self.boundary_points.copy()
        self.polygon = Polygon(
            np.array(
                [
                    self.exterior_points.real.reshape(-1),
                    self.exterior_points.imag.reshape(-1),
                ]
            ).T
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
        aaa_mmax=10,
        mirror_laurents=False,
        mirror_tol=0.5,
    ):
        """Create an interior curve from a parametric function."""
        side = self._generate_interior_curve_points(f, num_points)
        self._update_polygon()
        self._generate_interior_laurent_series(side, deg_laurent, centroid)
        if mirror_laurents:
            self._generate_mirror_laurents(side, deg_laurent, centroid, mirror_tol)
        if aaa:
            self._generate_aaa_poles(side, mmax=aaa_mmax)
        return side

    def add_mover(
        self,
        f,
        centroid,
        num_points=100,
        deg_laurent=10,
        aaa=False,
        mirror_laurents=False,
        mirror_tol=0.5,
    ):
        """Create an interior curve from a parametric function."""
        side = self.add_interior_curve(
            f=f,
            centroid=centroid,
            num_points=num_points,
            deg_laurent=deg_laurent,
            aaa=aaa,
            mirror_laurents=mirror_laurents,
            mirror_tol=mirror_tol,
        )
        self.movers += [side]
        return side

    def add_point(self, point):
        """Add a point to the domain."""
        self.boundary_points = np.append(self.boundary_points, point).reshape(-1, 1)
        side = f"{len(self.sides)}"
        self.error_points[side] = np.array(point)
        self.sides = np.append(self.sides, side)
        self.indices[side] = len(self.boundary_points) - 1

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
        points_to_delete = self.boundary_points[indices]
        for side, err_points in self.error_points.items():
            avg_diff = np.mean(
                np.abs(np.diff(self.boundary_points[self.indices[side]].T))
            )
            err_indices = np.where(
                np.abs(points_to_delete - err_points.reshape(-1)) <= avg_diff / 2
            )[1]
            self.error_points[side] = np.delete(err_points, err_indices)
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

    def dimensionalize(self, L):  # noqa N803
        """Assumes domain is symmetric about the origin."""
        self.boundary_points *= L
        self.corners *= L
        self.poles = [pole * L for pole in self.poles]
        self.exterior_laurents = [
            (centroid * L, degree) for centroid, degree in self.exterior_laurents
        ]
        self.interior_laurents = [
            (centroid * L, degree) for centroid, degree in self.interior_laurents
        ]
        self.centroids = {
            side: centroid * L for side, centroid in self.centroids.items()
        }
        self._update_polygon()

    def _generate_exterior_polygon_points(self):
        """Create a list of boundary points on each edge."""
        if self.spacing == "linear":
            spacing = np.linspace(0, 1, self.num_edge_points)
            error_spacing = np.linspace(0, 1, self.num_edge_points * 2)
        else:
            spacing = (np.tanh(np.linspace(-10, 10, self.num_edge_points)) + 1) / 2
            error_spacing = (
                np.tanh(np.linspace(-10, 10, self.num_edge_points * 2)) + 1
            ) / 2
        nc = len(self.corners)
        self.boundary_points = np.array(
            [
                self.corners[i]
                + (self.corners[(i + 1) % nc] - self.corners[i]) * spacing
                for i in range(len(self.corners))
            ]
        ).reshape(-1, 1)
        self.error_points = {
            str(i): self.corners[i]
            + (self.corners[(i + 1) % nc] - self.corners[i]) * error_spacing
            for i in range(len(self.corners))
        }
        self.sides = np.array([str(i) for i in range(len(self.corners))], dtype="<U50")
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
        error_points = f(np.linspace(0, 1, 2 * num_points)).astype(np.complex128)
        # create a shapely LineString and check it is simple
        # (i.e. does not intersect itself)
        line = LineString(np.array([points.real, points.imag]).T)
        if not line.is_simple:
            raise ValueError("Curve must not intersect itself")
        side = str(len(self.sides))
        self.sides = np.append(self.sides, side)
        self.interior_curves += [side]
        n_bp = len(self.boundary_points)
        self.indices[side] = [i for i in range(n_bp, n_bp + num_points)]
        self.boundary_points = np.concatenate(
            [self.boundary_points, points.reshape(-1, 1)], axis=0
        )
        self.error_points[side] = error_points
        return side

    def _name_side(self, old, new):
        """Rename the sides of the polygon."""
        if isinstance(old, str) and isinstance(new, str):
            if old not in self.sides:
                raise ValueError(f"Side {old} does not exist")
            if new in self.sides:
                raise ValueError(f"Side {new} already exists")
            self.sides[np.where(self.sides == old)] = new
            self.indices[new] = self.indices.pop(old)
            self.error_points[new] = self.error_points.pop(old)
        else:
            raise TypeError("Side names must be strings")

    def _group_sides(self, old_sides, new):
        """Rename a list of side labels as a single side label."""
        self.indices[str(new)] = []
        self.error_points[str(new)] = []
        for side in old_sides:
            self.sides = np.delete(self.sides, np.where(self.sides == side))
            self.indices[str(new)] += self.indices.pop(side)
            self.error_points[str(new)] = np.concatenate(
                [
                    self.error_points[str(new)],
                    self.error_points.pop(side),
                ]
            )
        self.sides = np.concatenate([self.sides, [str(new)]])

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
                / np.abs(self.corners[i] - self.corners[(i + 1) % len(self.corners)])
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
                            self.corners[(i + 1) % len(self.corners)] - self.corners[i]
                        ),
                    )
                )
                for i in range(len(self.corners))
            ]
        )
        self.poles = [
            self.corners[i] + signs[i] * bisectors[i] * pole_spacing
            for i in range(len(self.corners))
        ]

    def _get_centroid(self, side):
        """Calculate the centroid of a side."""
        points = self.boundary_points[self.indices[side]]
        return np.mean(points)

    def _generate_interior_laurent_series(self, side, degree, centroid):
        """Generate Laurent series in a hole of the domain."""
        if centroid is None:
            centroid = self._get_centroid(side)
        self.centroids[side] = centroid
        self.interior_laurents.append((centroid, degree))
        if side not in self.interior_laurent_indices.keys():
            self.interior_laurent_indices[side] = [len(self.interior_laurents) - 1]
        else:
            self.interior_laurent_indices[side] += [len(self.interior_laurents) - 1]

    def _generate_exterior_laurent_series(self, side, degree, centroid):
        """Generate Laurent series outside the domain."""
        self.exterior_laurents.append((centroid, degree))
        if side not in self.exterior_laurent_indices.keys():
            self.exterior_laurent_indices[side] = [len(self.exterior_laurents) - 1]
        else:
            self.exterior_laurent_indices[side] += [len(self.exterior_laurents) - 1]

    def _generate_mirror_laurents(self, side, degree, centroid, tol=2):
        """Generate mirror images of the Laurent series."""
        # for each side of the polygon, calculate the inwards
        # facing unit normal vector
        if centroid is None:
            centroid = self._get_centroid(side)
        n_corners = len(self.corners)
        corners = np.array(self.corners)
        normals = np.zeros((n_corners), dtype=np.complex128)
        tangents = self.corners - np.roll(self.corners, 1)
        normals = 1j * tangents
        normals = normals / np.abs(normals)
        x2, y2 = corners.real, corners.imag
        x1, y1 = np.roll(corners, 1).real, np.roll(corners, 1).imag
        x0, y0 = centroid.real, centroid.imag
        distances = np.abs((x2 - x1) * (y1 - y0) + (x1 - x0) * (y2 - y1)) / np.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )
        mirrors = centroid - 2 * normals * distances
        for mirror in mirrors:
            if np.abs(mirror - centroid) < tol:
                self._generate_exterior_laurent_series(side, degree, mirror)
                if side not in self.mirror_indices.keys():
                    self.mirror_indices[side] = [len(self.exterior_laurents) - 1]
                else:
                    self.mirror_indices[side] += [len(self.exterior_laurents) - 1]

    def _generate_aaa_poles(self, side, mmax=None):
        """Generate aaa poles that lie outside the domain."""
        z = self.boundary_points[self.indices[side]][:-1]
        f = np.conj(z)
        if mmax is None:
            _, poles, _, _ = aaa(f, z)
        else:
            _, poles, _, _ = aaa(f, z, mmax=mmax)
        exterior_poles = [pole for pole in poles if not self.__contains__(pole)]
        self.poles = list(self.poles) + [np.array(exterior_poles)]

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
            np.array(
                [
                    self.exterior_points.real.reshape(-1),
                    self.exterior_points.imag.reshape(-1),
                ]
            ).T,
            holes=holes,
        )
        if buffer > 0:
            self.polygon = self.polygon.buffer(buffer)

    def plot(self, figax=None, set_lims=True, point_color="indices", legend=True):
        """Display the labelled polygon."""
        if figax is None:
            fig, ax = plt.subplots()
        flat_poles = np.hstack([pole_group.flatten() for pole_group in self.poles])
        if self.exterior_laurents:
            flat_laurents = np.hstack([laurent[0] for laurent in self.exterior_laurents])
            flat_poles = np.hstack([flat_poles, flat_laurents])
        if set_lims:
            try:
                x_min = min(flat_poles.real)
                x_max = max(flat_poles.real)
                y_min = min(flat_poles.imag)
                y_max = max(flat_poles.imag)
            except ValueError:
                x_min = min(self.exterior_points.real)
                x_max = max(self.exterior_points.real)
                y_min = min(self.exterior_points.imag)
                y_max = max(self.exterior_points.imag)
            x_min = min(min(self.exterior_points.real)[0], x_min)
            x_max = max(max(self.exterior_points.real)[0], x_max)
            y_min = min(min(self.exterior_points.imag)[0], y_min)
            y_max = max(max(self.exterior_points.imag)[0], y_max)
            ax.set_xlim(x_min - 0.1, x_max + 0.1)
            ax.set_ylim(y_min - 0.1, y_max + 0.1)
        self.plot_polygon(ax, self.polygon)
        for side in self.sides:
            bp = self.boundary_points[self.indices[side]]
            mid_point = bp[len(bp) // 2]
            x, y = mid_point.real, mid_point.imag
            ax.text(
                x,
                y,
                str(side),
                fontsize=20,
                ha="left",
                va="center",
            )

        total_points = len(self.boundary_points)
        if point_color == "indices":
            color = np.arange(total_points)
        else:
            color = "k"
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
        if legend:
            ax.legend(
                handles=handles,
                labels=["Lightning poles"] + degree_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1),
            )
        ax.set_aspect("equal")
        ax.axis("off")
        # plt.tight_layout()
        return fig, ax

    def plot_polygon(
        self,
        ax,
        poly,
        exterior_color="white",
        interior_color=None,
        zorder=1,
    ):
        """Plot a polygon on the given axis."""
        exterior_coords = np.array(self.polygon.exterior.coords)
        exterior_patch = patches.Polygon(
            exterior_coords, color=interior_color, edgecolor="k"
        )
        ax.add_patch(exterior_patch)
        for inner in poly.interiors:
            interior_patch = patches.Polygon(
                np.array(inner.coords),
                facecolor=exterior_color,
                edgecolor="k",
                zorder=zorder,
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

    def exterior_distance(self, point):
        """Determine the distance from a point to the exterior of the domain."""
        if isinstance(point, complex):
            point = Point(np.array([point.real, point.imag]))
            return self.polygon.exterior.distance(point)
        else:
            return TypeError("point must be a complex number")

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
            # return self.polygon.contains(
            #     point
            # ) or self.polygon.exterior.contains(point)
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
