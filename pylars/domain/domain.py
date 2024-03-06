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
        deg_poly=24,
        num_edge_points=100,
        num_poles=24,
        sigma=4,
        length_scale=1,
        spacing="clustered",
    ):
        self.boundary_points = np.empty((0, 1))
        self.error_points = {}
        self.weights = np.empty((0, 1))
        self.sides = np.empty((0), dtype="<U50")
        self.indices = {}
        self.interior_curves = []
        self.centroids = {}
        self.movers = []
        self.deg_poly = deg_poly
        self.poles = []
        self.interior_laurents = []
        self.exterior_laurents = []
        self.interior_laurent_indices = {}
        self.exterior_laurent_indices = {}
        self.mirror_indices = {}
        self.add_exterior_polygon(
            corners, num_edge_points, num_poles, sigma, length_scale, spacing
        )

    def add_exterior_polygon(
        self,
        corners,
        num_edge_points=100,
        num_poles=24,
        sigma=4,
        length_scale=1,
        spacing="clustered",
    ):
        self.corners = np.array(corners)
        self._generate_exterior_polygon_points(
            corners=corners, num_edge_points=num_edge_points, spacing_type=spacing
        )
        self._generate_lightning_poles(corners, num_poles, sigma, length_scale)
        self.polygon = Polygon(
            np.array(
                [
                    self.exterior_points.real.reshape(-1),
                    self.exterior_points.imag.reshape(-1),
                ]
            ).T
        )

    def add_interior_polygon(
        self,
        corners,
        num_edge_points=100,
        num_poles=24,
        sigma=4,
        length_scale=1,
        spacing="clustered",
    ):
        self._generate_interior_polygon_points(
            corners=corners, num_edge_points=num_edge_points, spacing_type=spacing
        )
        self._generate_lightning_poles(corners[::-1], num_poles, sigma, length_scale)
        self._update_polygon()

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

    def add_inerior_polygon(
        self,
        corners,
        num_points=100,
        spacing="clustered",
        lightining_poles=True,
        length_scale=1,
        sigma=4,
        aaa=False,
        aaa_mmax=10,
    ):
        """Create an interior curve from a parametric function."""
        side = self._generate_interior_polygon_points(corners, num_points, spacing)
        centroid = np.mean(corners)
        self._update_polygon()
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

    def _generate_exterior_polygon_points(self, corners, num_edge_points, spacing_type):
        if spacing_type == "linear":
            spacing = np.linspace(0, 1, num_edge_points)
        else:
            spacing = (np.tanh(np.linspace(-10, 10, num_edge_points)) + 1) / 2
        error_spacing = np.linspace(0, 1, num_edge_points * 2)
        nc = len(corners)
        generated_points = np.empty((0, 1))
        for i in range(nc):
            points = corners[i] + (corners[(i + 1) % nc] - corners[i]) * spacing
            points = np.array(points).reshape(-1, 1)
            self.boundary_points = np.concatenate([self.boundary_points, points])
            generated_points = np.concatenate([generated_points, points])
            error_points = (
                corners[i] + (corners[(i + 1) % nc] - corners[i]) * error_spacing
            )
            self.error_points[str(i)] = np.array(error_points)
            self.sides = np.append(self.sides, i)
            self.indices[str(i)] = [
                j for j in range(i * num_edge_points, (i + 1) * num_edge_points)
            ]
        self.exterior_points = generated_points

    def _generate_interior_polygon_points(self, corners, num_edge_points, spacing_type):
        side = str(len(self.sides))
        self.sides = np.append(self.sides, side)
        n_bp = len(self.boundary_points)
        num_points = len(corners) * num_edge_points
        self.indices[side] = [i for i in range(n_bp, n_bp + num_points)]
        self.interior_curves += [side]
        self.centroids[side] = np.mean(corners)
        nc = len(corners)
        if spacing_type == "linear":
            spacing = np.linspace(0, 1, num_edge_points)
        else:
            spacing = (np.tanh(np.linspace(-10, 10, num_edge_points)) + 1) / 2
        error_spacing = np.linspace(0, 1, num_edge_points * 2)
        nc = len(corners)
        for i in range(nc):
            points = corners[i] + (corners[(i + 1) % nc] - corners[i]) * spacing
            points = np.array(points).reshape(-1, 1)
            self.boundary_points = np.concatenate([self.boundary_points, points])
            error_points = (
                corners[i] + (corners[(i + 1) % nc] - corners[i]) * error_spacing
            )
        self.error_points[side] = np.array(error_points)

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
        n_bp = len(self.boundary_points)
        self.sides = np.append(self.sides, side)
        self.indices[side] = [i for i in range(n_bp, n_bp + num_points)]
        self.interior_curves += [side]
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

    def _generate_lightning_poles(self, corners, num_poles, sigma, length_scale):
        """Generate exponentially clustered lightning poles.

        Poles are clustered exponentially. If the corners are positively oriented,
        returns exterior poles. If negatively oriented, returns interior poles.
        """
        pole_spacing = cluster(num_poles, length_scale, sigma)
        nc = len(corners)
        # find the exterior angle bisector at each corner
        for i in range(nc):
            left = corners[i] - corners[i - 1]
            right = corners[(i + 1) % nc] - corners[i]
            bisector = left / np.abs(left) - right / np.abs(right)
            bisector /= np.abs(bisector)
            # make an array of length nc which is one
            # if the midpoint of the line joining the two neighbouring corners
            # is inside the domain and -1 otherwise
            sign = np.sign(
                np.cross(
                    cart(left),
                    cart(right),
                )
            )
            self.poles = list(self.poles) + [
                corners[i] + sign * bisector * pole_spacing
            ]

    def _generate_clustered_poles(
        self, num_poles, location, direction, length_scale=1, sigma=4
    ):
        pole_spacing = cluster(num_poles, length_scale, sigma)
        self.poles = list(self.poles) + [location + direction * pole_spacing]

    def _generate_pole_ring(self, num_poles, radius, center):
        thetas = np.linspace(0, 2 * np.pi, num_poles, endpoint=False)
        self.poles = list(self.poles) + [center + radius * np.exp(1j * thetas)]

    def _get_centroid(self, side):
        """Calculate the centroid of a side."""
        points = self.boundary_points[self.indices[side]]
        return np.mean(points)

    def _generate_interior_laurent_series(self, side, degree, centroid):
        """Generate Laurent series in a hole of the domain."""
        if centroid is None:
            centroid = self._get_centroid(side)
        self.centroids[side] = centroid
        self.interior_laurents.append((np.complex128(centroid), np.int64(degree)))
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

    def _generate_aaa_poles(self, side=None, mmax=None):
        """Generate aaa poles that lie outside the domain."""
        if side is None:
            z = self.boundary_points[:-1]
        else:
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
            flat_laurents = np.hstack(
                [laurent[0] for laurent in self.exterior_laurents]
            )
            outer_most = np.hstack([flat_poles, flat_laurents])
        else:
            outer_most = flat_poles
        if set_lims:
            try:
                x_min = min(outer_most.real)
                x_max = max(outer_most.real)
                y_min = min(outer_most.imag)
                y_max = max(outer_most.imag)
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
        interior_degrees = []
        exterior_degrees = []
        degree_labels = []
        for centroid, degree in self.interior_laurents:
            laruent = ax.scatter(
                centroid.real,
                centroid.imag,
                c=[(0, (1 - np.exp(-degree / 10)), 0)],
                s=10,
            )
            if degree not in interior_degrees:
                interior_degrees.append(degree)
                handles.append(laruent)
                degree_labels.append(f"Laurent series ({degree}) with log")
        for centroid, degree in self.exterior_laurents:
            laruent = ax.scatter(
                centroid.real,
                centroid.imag,
                c=[(0, (1 - np.exp(-degree / 10)), 0)],
                s=10,
            )
            if degree not in exterior_degrees:
                exterior_degrees.append(degree)
                handles.append(laruent)
                degree_labels.append(f"Laurent series ({degree})")
        if legend:
            ax.legend(
                handles=handles,
                labels=["Lightning poles"] + degree_labels,
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
