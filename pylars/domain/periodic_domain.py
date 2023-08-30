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
        corners = np.array(
            [
                length / 2 + 1j * height / 2,
                -length / 2 + 1j * height / 2,
                -length / 2 - 1j * height / 2,
                length / 2 - 1j * height / 2,
            ]
        )
        self.rect = Polygon(np.array([corners.real, corners.imag]).T)
        super().__init__(
            corners,
            num_edge_points,
            num_poles,
            sigma,
            length_scale,
            deg_poly,
            spacing,
        )
        self.image_indices = {}
        self.periodicity = periodicity
        self.periodic_curves = []

    def add_periodic_curve(
        self,
        f,
        centroid,
        num_points=100,
        deg_laurent=10,
        aaa=False,
        aaa_mmax=None,
        mirror_laurents=False,
        mirror_tol=0.5,
        image_laurents=False,
        image_tol=0.5,
    ):
        """Create a periodic curve from a parametric function."""
        if self.periodicity is None:
            raise ValueError("Periodicity must be specified")
        side = str(len(self.sides))
        self.sides = np.append(self.sides, side)
        self.periodic_curves += [side]
        self._generate_intersecting_images(
            side, f, num_points, centroid, deg_laurent
        )
        self._update_polygon()
        if mirror_laurents:
            self._generate_mirror_laurents(
                side, deg_laurent, centroid, mirror_tol
            )
        if image_laurents:
            self._generate_image_laurents(
                side, deg_laurent, centroid, image_tol
            )
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
        image_laurents=False,
        image_tol=0.5,
    ):
        """Create an interior curve from a parametric function."""
        side = self.add_periodic_curve(
            f,
            centroid,
            num_points,
            deg_laurent,
            aaa,
            mirror_laurents,
            mirror_tol,
            image_laurents,
            image_tol,
        )
        self.movers += [side]
        return side

    def get_nn_image_centroids(self, centroid, original=True):
        """Generate the nearest neighbour image centroids."""
        l, h = self.length, self.height
        disp = np.array(
            [
                [-l - 1j * h, -1j * h, l - 1j * h],
                [-l, 0, l],
                [-l + 1j * h, 1j * h, l + 1j * h],
            ]
        ).flatten()
        if not original:
            disp = np.delete(disp, 4)
        nnic = centroid + disp
        return nnic

    def _generate_image_laurents(self, side, deg_laurent, centroid, tol=0.8):
        """Generate the image laurents."""
        if centroid is None:
            centroid = self._get_centroid(side)
        nnic = self.get_nn_image_centroids(centroid, original=False)
        for image in nnic:
            point = Point(np.array([image.real, image.imag]))
            if self.rect.exterior.distance(point) < tol:
                self._generate_exterior_laurent_series(
                    side, deg_laurent, image
                )
                if side not in self.image_indices.keys():
                    self.image_indices[side] = [
                        len(self.exterior_laurents) - 1
                    ]
                else:
                    self.image_indices[side] += [
                        len(self.exterior_laurents) - 1
                    ]

    def _generate_intersecting_images(
        self, side, curve, num_points, centroid, deg_laurent
    ):
        """Get the images of a curve that intersect the domain."""
        # TODO make sure the remaining rectangle points are the
        # same on each side
        # TODO fix case before points are added.
        if centroid is None:
            points = curve(np.linspace(0, 1, num_points))
            centroid = np.mean(points)
        nnic = self.get_nn_image_centroids(centroid)
        # TODO add error_points
        centered_points = curve(np.linspace(0, 1, num_points)) - centroid
        n_bp = len(self.boundary_points)
        self.indices[side] = None
        for image in nnic:
            translated_points = centered_points + image
            intersecting_inds = np.where(
                (np.abs(translated_points.real) < self.length / 2)
                & (np.abs(translated_points.imag) < self.height / 2)
            )
            if len(intersecting_inds[0]) == len(translated_points):
                # add the whole curve if fully contained
                self._generate_interior_laurent_series(
                    side, deg_laurent, image
                )
                self.indices[side] = [
                    i for i in range(n_bp, n_bp + num_points)
                ]
                self.boundary_points = np.concatenate(
                    [self.boundary_points, translated_points.reshape(-1, 1)],
                    axis=0,
                )
                self.interior_curves += [side]
                break
            if len(intersecting_inds[0]) > 0:
                self._generate_exterior_laurent_series(
                    side, deg_laurent, image
                )
                rect_inds = np.concatenate(
                    [self.indices[str(i)] for i in range(4)]
                )
                rect_points = self.boundary_points[rect_inds]
                trans_poly_points = np.array(
                    [translated_points.real, translated_points.imag]
                ).T
                poly = Polygon(trans_poly_points)
                in_curve_indices = np.array(
                    [
                        ind
                        for ind, rect_point in enumerate(rect_points)
                        if poly.contains(
                            Point([rect_point.real, rect_point.imag])
                        )
                    ]
                )
                # intersecting points may indices may jump over the end
                # of the array, fixing this
                if not np.all(np.diff(intersecting_inds) == 1):
                    jump = (
                        np.where(np.diff(intersecting_inds)[0] != 1)[0][0] + 1
                    )
                    intersecting_inds = np.concatenate(
                        [
                            intersecting_inds[0][jump:] - num_points,
                            intersecting_inds[0][:jump],
                        ]
                    )
                intersecting_points = np.array(
                    translated_points[intersecting_inds]
                ).reshape(-1, 1)

                n_bp = len(self.boundary_points)
                new_indices = [
                    int(i)
                    for i in range(n_bp, n_bp + len(intersecting_points))
                ]
                if self.indices[side] is None:
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
                if not np.all(np.diff(in_curve_indices) == 1):
                    jump = np.where(np.diff(in_curve_indices) != 1)[0][0] + 1
                    in_curve_indices = np.concatenate(
                        [
                            in_curve_indices[jump:] - len(rect_points),
                            in_curve_indices[:jump],
                        ]
                    )
                first_rect_point = rect_points[in_curve_indices[0]]
                first_rect_ind = np.where(
                    self.exterior_points == first_rect_point
                )[0][0]
                last_rect_point = rect_points[
                    (in_curve_indices[-1]) % len(rect_points)
                ]
                last_rect_ind = np.where(
                    self.exterior_points == last_rect_point
                )[0][0]
                if first_rect_ind >= last_rect_ind:
                    after = self.exterior_points[last_rect_ind:first_rect_ind]
                    self.exterior_points = np.concatenate(
                        [intersecting_points[::-1], after]
                    )
                else:
                    before = self.exterior_points[:first_rect_ind]
                    after = self.exterior_points[last_rect_ind:]
                    self.exterior_points = np.concatenate(
                        [before, intersecting_points[::-1], after]
                    )
                # TODO this assumes that the curve is convex?
                # only intersects once.
                # intersecting_points must be inverted as non-convex segments
                # move clockwise
                if np.any(in_curve_indices):
                    in_curve_indices = np.sort(
                        in_curve_indices % len(rect_points)
                    )
                    self.remove(in_curve_indices)
