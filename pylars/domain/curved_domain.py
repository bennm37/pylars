"""Class for creating curved domains."""
import numpy as np
from pylars import Domain
from pylars.numerics import aaa as aaa_func
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
        self.curve = curve
        self.num_edge_points = num_edge_points
        self.deg_poly = deg_poly
        self.spacing = spacing
        self._generate_exterior_curve_points(curve, num_edge_points)
        if aaa:
            self._generate_exterior_aaa_poles(aaa_mmax)
        else:
            self.poles = np.array([[]])
            self.num_poles = len(self.poles)
        self.interior_curves = []
        self.centroids = {}
        self.movers = []
        self.interior_laurents = []
        self.exterior_laurents = []
        self.interior_laurent_indices = {}
        self.exterior_laurent_indices = {}
        self.mirror_indices = {}
        self._update_polygon()

    def _generate_exterior_curve_points(self, f, num_points):
        if not np.isclose(f(0), f(1)):
            raise ValueError("Curve must be closed")
        points = f(np.linspace(0, 1, num_points)).astype(np.complex128)
        error_points = f(np.linspace(0, 1, 2 * num_points)).astype(
            np.complex128
        )
        # create a shapely LineString and check it is simple
        # (i.e. does not intersect itself)
        line = LineString(np.array([points.real, points.imag]).T)
        if not line.is_simple:
            raise ValueError("Curve must not intersect itself")
        side = "0"
        self.sides = [side]
        self.exterior_curves = [side]
        self.indices = {side: [i for i in range(num_points)]}
        self.boundary_points = points.reshape(-1, 1)
        self.error_points = {side: error_points}
        self.exterior_points = self.boundary_points.copy()
        return side

    def _generate_exterior_aaa_poles(self, mmax=100):
        z = self.exterior_points
        f = np.conj(z)
        if mmax is None:
            _, poles, _, _ = aaa_func(f, z)
        else:
            _, poles, _, _ = aaa_func(f, z, mmax=mmax)
        exterior_poles = [
            pole for pole in poles if not self.__contains__(pole)
        ]
        self.poles = np.array(exterior_poles)
        self.num_poles = len(self.poles)
