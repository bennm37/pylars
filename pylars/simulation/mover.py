"""Mover class for moving objects."""
import numpy as np
from shapely import Point, Polygon
import copy


class Mover:
    """Class for adding moving objects in simulations."""

    def __init__(
        self,
        curve,
        deriv,
        centroid,
        angle=0,
        velocity=0,
        angular_velocity=0,
        density=1,
    ):
        self.initial_curve = lambda t: (curve(t) - centroid) * np.exp(
            -1j * angle
        )
        self.initial_deriv = lambda t: deriv(t) * np.exp(-1j * angle)
        self.curve = curve
        self.deriv = deriv
        self.angle = np.float64(angle)
        self.centroid = np.complex128(centroid)
        self.velocity = np.complex128(velocity)
        self.angular_velocity = np.float64(angular_velocity)
        self.density = density

    def set_mass(self):
        """Calculate the mass of the mover."""
        # approximate the area of the mover
        points = self.curve(np.linspace(0, 1, 500))
        points = np.array([points.real, points.imag]).T
        self.poly = Polygon(points)
        self.area = self.poly.area
        mass_density = lambda x, y: self.density * np.ones_like(x)
        self.mass = self.mc_integrate(mass_density)
        # assert np.isclose(self.density * self.area, self.mass)
        moi_density = lambda x, y: self.density * (x - self.centroid)
        self.moi = self.mc_integrate(self.poly, moi_density)
        self.mass = self.density * self.area
        self.moi = self.mass

    def mc_integrate(self, integrand, num_samples=10000):
        """Monte Carlo integration of a function over a polygon."""
        points_x = np.random.uniform(
            self.poly.bounds[0], self.poly.bounds[2], num_samples
        )
        points_y = np.random.uniform(
            self.poly.bounds[1], self.poly.bounds[3], num_samples
        )
        bounding_area = (self.poly.bounds[2] - self.poly.bounds[0]) * (
            self.poly.bounds[3] - self.poly.bounds[1]
        )
        points = np.array([points_x, points_y]).T
        points = [p for p in points if self.poly.contains(Point(p))]
        points = np.array(points)
        return (
            bounding_area
            * np.sum(integrand(points[:, 0], points[:, 1]))
            / num_samples
        )

    def translate(self, disp):
        """Translate the mover."""
        self.centroid += disp
        self.curve = lambda t: self.centroid + self.initial_curve(t) * np.exp(
            1j * self.angle
        )
        self.deriv = lambda t: self.initial_deriv(t) * np.exp(1j * self.angle)

    def rotate(self, angle):
        """Rotate the mover."""
        self.angle += angle
        self.curve = lambda t: self.centroid + self.initial_curve(t) * np.exp(
            1j * self.angle
        )
        self.deriv = lambda t: self.initial_deriv(t) * np.exp(1j * self.angle)

    def copy(self):
        """Create a deepcopy."""
        return copy.deepcopy(self)
