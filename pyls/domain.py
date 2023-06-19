"""Create a polygonal domain from a list of corners.

Raises:
    ValueError: Domain has less than 3 corners
    TypeError: Corners must be a list of complex numbers
    TypeError: Number of boundary points must be a positive integer
"""
import numpy as np  # noqa: D100
import matplotlib.pyplot as plt
from numbers import Number
from pyls.numerics import cart


class Domain:
    """Create a polygonal domain from a list of corners."""

    def __init__(
        self, corners, num_boundary_points=100, num_poles=24, sigma=4
    ):
        self.corners = np.array(corners)
        self.num_boundary_points = num_boundary_points
        self.num_poles = num_poles
        self.sigma = sigma
        self.check_input()
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
        if type(self.num_poles) != int or self.num_poles <= 0:
            raise TypeError("num_poles must be a positive integer")

    def generate_boundary_points(self):
        """Create a list of boundary points on each edge.

        Points are clustered towards the corners.
        """
        spacing = (
            np.tanh(np.linspace(-16, 16, self.num_boundary_points)) + 1
        ) / 2
        self.boundary_points = np.array(
            [
                self.corners[i - 1]
                + (self.corners[i] - self.corners[i - 1]) * spacing
                for i in range(len(self.corners))
            ]
        ).flatten()

    def generate_poles(self):
        """Generate exponentially clustered lighting poles.

        Poles are clustered using"""
        pole_spacing = np.exp(
            self.sigma
            * (np.sqrt(range(self.num_poles, 1, -1)) - np.sqrt(self.num_poles))
        )
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
        ).flatten()

    def show(self):
        """Display the labelled polygon."""
        fig, ax = plt.subplots()
        x_min = min(self.poles.real)
        x_max = max(self.poles.real)
        y_min = min(self.poles.imag)
        y_max = max(self.poles.imag)
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)
        cartesian_corners = np.array([self.corners.real, self.corners.imag]).T
        polygon = plt.Polygon(cartesian_corners, fill=True, alpha=0.5)
        ax.add_patch(polygon)
        # label each edge of the polygon in order
        for i in range(len(self.corners)):
            x = (self.corners[i].real + self.corners[i - 1].real) / 2
            y = (self.corners[i].imag + self.corners[i - 1].imag) / 2
            ax.text(
                x,
                y,
                str(i),
                fontsize=20,
                ha="center",
                va="center",
            )
        total_points = self.num_boundary_points * len(self.corners)
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
            self.poles.real,
            self.poles.imag,
            c="red",
            s=10,
        )
        ax.set_aspect("equal")
        plt.show()
