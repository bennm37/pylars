"""Class for creating curved domains."""
import numpy as np
from pylars import Domain
from shapely import LineString

# from pylars.numerics import aaa as aaa_func
# from numbers import Number, Integral
# import matplotlib.patches as patches


class CurvedDomain(Domain):
    """Class for creating curved domains."""

    def __init__(
        self,
        curve,
        num_edge_points=500,
        aaa=False,
        aaa_mmax=None,
        deg_poly=30,
        spacing="linear",
    ):
        self.corners = np.array([])
        self.num_edge_points = num_edge_points
        self.deg_poly = deg_poly
        self.spacing = spacing
        self.check_input()
        self._generate_exterior_curve_points(num_edge_points)
        self._generate_exterior_aaa_poles(aaa, aaa_mmax)

        self.interior_curves = []
        self.centroids = {}
        self.movers = []
        self.interior_laurents = []
        self.exterior_laurents = []
        self.interior_laurent_indices = {}
        self.exterior_laurent_indices = {}
        self.mirror_indices = {}

    def _generate_exterior_curve_points(self, f, num_points):
        if not np.isclose(f(0), f(1)):
            raise ValueError("Curve must be closed")
        points = f(np.linspace(0, 1, num_points)).astype(np.complex128)
        # create a shapely LineString and check it is simple
        # (i.e. does not intersect itself)
        line = LineString(np.array([points.real, points.imag]).T)
        if not line.is_simple:
            raise ValueError("Curve must not intersect itself")
        side = str(len(self.sides))
        self.sides = np.append(self.sides, side)
        self.exterior_curves += [side]
        n_bp = len(self.boundary_points)
        self.indices[side] = [i for i in range(n_bp, n_bp + num_points)]
        self.boundary_points = np.concatenate(
            [self.boundary_points, points.reshape(-1, 1)], axis=0
        )
        return side

    def _generate_exterior_aaa_poles(self):
        pass
