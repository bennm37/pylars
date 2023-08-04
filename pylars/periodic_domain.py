"""Child class for creating periodic domains."""
from pylars import Domain
import numpy as np
from shapely import Polygon, Point


class PeriodicDomain(Domain):
    """Class for creating periodic domains.

    Currently only rectangular domains are supported.
    """

    def __init__(
        self,
        length,
        height,
        num_edge_points=100,
        num_poles=0,
        sigma=0,
        length_scale=0,
        deg_poly=10,
        spacing="linear",
        periodicity="xy",
    ):
        # anticlockwise from top left
        self.length = length
        self.height = height
        corners = [
            length / 2 + 1j * height / 2,
            -length / 2 + 1j * height / 2,
            -length / 2 - 1j * height / 2,
            length / 2 - 1j * height / 2,
        ]
        super().__init__(
            corners,
            num_edge_points,
            num_poles,
            sigma,
            length_scale,
            deg_poly,
            spacing,
        )
        self.periodicity = periodicity
        self.periodic_curves = []

    def add_periodic_curve(
        self,
        f,
        centroid,
        num_points=100,
        deg_laurent=10,
    ):
        """Create a periodic curve from a parametric function."""
        if self.periodicity is None:
            raise ValueError("Periodicity must be specified")
        side = str(len(self.sides))
        self.sides = np.concatenate([self.sides, [side]])
        self.periodic_curves += [side]
        self._generate_intersecting_images(
            side, f, num_points, centroid, deg_laurent
        )
        self._update_polygon()

    def get_nn_image_centroids(self, centroid):
        """Generate the nearest neighbour image centroids."""
        l, h = self.length, self.height
        disp = np.array(
            [
                [-l - 1j * h, -1j * h, l - 1j * h],
                [-l, 0, l],
                [-l + 1j * h, 1j * h, l + 1j * h],
            ]
        ).flatten()
        self.nnic = centroid + disp

    def _generate_intersecting_images(
        self, side, curve, num_points, centroid, deg_laurent
    ):
        """Get the images of a curve that intersect the domain."""
        # TODO make sure the remaining rectangle points are the
        # same on each side
        # TODO fix case before points are added.
        self.get_nn_image_centroids(centroid)
        centered_points = curve(np.linspace(0, 1, num_points)) - centroid
        n_bp = len(self.boundary_points)
        original = n_bp - 1
        self.indices[side] = [original]
        for c in self.nnic:
            translated_points = centered_points + c
            intersecting_inds = np.where(
                (np.abs(translated_points.real) < self.length / 2)
                & (np.abs(translated_points.imag) < self.height / 2)
            )
            if len(intersecting_inds) > 0:
                # get the rectangle points inside the curve
                self.laurents.append((c, deg_laurent))
                rect_inds = np.concatenate(
                    [self.indices[str(i)] for i in range(4)]
                )
                rect_points = self.boundary_points[rect_inds]
                # remove the boundary points inside the curve
                trans_poly_points = np.array(
                    [translated_points.real, translated_points.imag]
                ).T
                poly = Polygon(trans_poly_points)
                in_curve_indices = [
                    ind
                    for ind, rect_point in enumerate(rect_points)
                    if poly.contains(Point([rect_point.real, rect_point.imag]))
                ]
                self.remove(in_curve_indices)
                # add the intersecting points to the side
                intersecting_points = np.array(
                    translated_points[intersecting_inds]
                ).reshape(-1, 1)
                n_bp = len(self.boundary_points)
                new_indices = [
                    int(i)
                    for i in range(n_bp, n_bp + len(intersecting_points))
                ]
                if (
                    len(self.indices[side]) == 1
                    and self.indices[side][0] == original
                ):
                    self.indices[side] = new_indices
                else:
                    self.indices[side] = np.concatenate(
                        [self.indices[side], new_indices]
                    ).astype(int)
                self.boundary_points = np.concatenate(
                    [
                        self.boundary_points,
                        intersecting_points,
                    ]
                )
