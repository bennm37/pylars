"""Module defining the Problem Class."""
from pylars import Domain, PeriodicDomain, CurvedDomain
from collections.abc import Sequence
from time import perf_counter
import numpy as np
import copy

OPERATIONS = ["[::-1]", "+", "-", "*", "/", "**", "(", ")"]
DEPENDENT = ["psi", "u", "v", "p", "e11", "e12"]
INDEPENDENT = ["x", "y"]


class Problem:
    """Class for setting up Domains and Boundary conditions."""

    def __init__(self, domain=None, boundary_conditons=None):
        self.creation_time = perf_counter()
        self.domain = domain
        self.boundary_conditions = boundary_conditons
        self.domain_type = None
        self.status = "nd"

    def add_exterior_polygon(
        self,
        corners,
        num_edge_points=100,
        length_scale=1,
        sigma=1,
        deg_poly=10,
        num_poles=10,
        spacing="clustered",
    ):
        """Create a new Domain object from a list of corners."""
        if self.domain is not None:
            raise Warning("Deleting old domain object and creating a new one.")
        self.domain = Domain(
            corners=corners,
            num_edge_points=num_edge_points,
            num_poles=num_poles,
            length_scale=length_scale,
            sigma=sigma,
            deg_poly=deg_poly,
            spacing=spacing,
        )
        self.domain_type = "polygon"

    def add_periodic_domain(
        self,
        length,
        height,
        num_edge_points=100,
        length_scale=0,
        sigma=0,
        deg_poly=20,
        num_poles=0,
        spacing="linear",
    ):
        """Create a PeriodicDomain object from its aspect ratio."""
        if self.domain is not None:
            raise Warning("Deleting old domain object and creating a new one.")
        self.domain = PeriodicDomain(
            length=length,
            height=height,
            num_edge_points=num_edge_points,
            num_poles=num_poles,
            length_scale=length_scale,
            sigma=sigma,
            deg_poly=deg_poly,
            spacing=spacing,
        )
        self.domain_type = "periodic"

    def add_curved_domain(
        self,
        curve,
        num_edge_points=100,
        deg_poly=20,
        aaa=False,
        aaa_mmax=None,
        spacing="linear",
    ):
        """Add a curved domain."""
        if self.domain is not None:
            raise Warning("Deleting old domain object and creating a new one.")
        self.domain = CurvedDomain(
            curve,
            num_edge_points=num_edge_points,
            aaa=aaa,
            aaa_mmax=aaa_mmax,
            deg_poly=deg_poly,
            spacing=spacing,
        )
        self.domain_type = "curved"

    def add_interior_curve(
        self,
        f,
        num_points=100,
        deg_laurent=10,
        centroid=None,
        aaa=False,
        aaa_mmax=None,
        mirror_laurents=False,
        mirror_tol=0.5,
    ):
        """Create an interior curve from a parametric function.

        f(t) should be a closed simply connected parametric curve
        defined on t in [0,1].
        """
        if self.domain_type == "periodic":
            raise ValueError(
                """Can't add interior curve to periodic domain.
                Use add_periodic_curve instead."""
            )
        if self.domain is None:
            raise ValueError("Exterior polygon must be set first.")
        self.domain.add_interior_curve(
            f=f,
            num_points=num_points,
            deg_laurent=deg_laurent,
            centroid=centroid,
            aaa=aaa,
            aaa_mmax=aaa_mmax,
            mirror_laurents=mirror_laurents,
            mirror_tol=mirror_tol,
        )

    def add_periodic_curve(
        self,
        f,
        centroid=None,
        num_points=100,
        deg_laurent=10,
        aaa=False,
        aaa_mmax=None,
        image_laurents=False,
        image_tol=0.5,
        mirror_laurents=False,
        mirror_tol=0.5,
    ):
        """Create an interior curve from a parametric function.

        f(t) should be a closed simply connected parametric curve
        defined on t in [0,1].
        """
        if self.domain_type == "polygon":
            raise ValueError(
                """Can't add periodic curve to polygonal domain.
                Use add_interior_curve instead."""
            )
        if self.domain is None:
            raise ValueError("Periodic domain must be set first.")
        self.domain.add_periodic_curve(
            f=f,
            num_points=num_points,
            deg_laurent=deg_laurent,
            centroid=centroid,
            aaa=aaa,
            aaa_mmax=aaa_mmax,
            mirror_laurents=mirror_laurents,
            mirror_tol=mirror_tol,
            image_laurents=image_laurents,
            image_tol=image_tol,
        )

    def add_mover(
        self,
        mover,
        num_points=100,
        deg_laurent=10,
        aaa=False,
        mirror_laurents=False,
        mirror_tol=0.5,
        image_laurents=False,
        image_tol=0.5,
    ):
        """Add a mover object.

        f(t) should be a closed simply connected parametric curve
        defined on t in [0,1].
        """
        if aaa:
            return NotImplemented
        if self.domain is None:
            raise ValueError("Exterior polygon must be set first.")
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                side: None for side in self.domain.sides
            }
        if not hasattr(self, "error_values"):
            self.error_values = {}
        if isinstance(self.domain, PeriodicDomain):
            side = self.domain.add_mover(
                f=mover.curve,
                num_points=num_points,
                deg_laurent=deg_laurent,
                centroid=mover.centroid,
                mirror_laurents=mirror_laurents,
                mirror_tol=mirror_tol,
                image_laurents=image_laurents,
                image_tol=image_tol,
            )
        else:
            side = self.domain.add_mover(
                f=mover.curve,
                num_points=num_points,
                deg_laurent=deg_laurent,
                centroid=mover.centroid,
                mirror_laurents=mirror_laurents,
                mirror_tol=mirror_tol,
            )
        diff = mover.curve(np.linspace(0, 1, num_points)) - mover.centroid
        rs = np.abs(diff)
        thetas = np.angle(diff)
        error_diff = (
            mover.curve(np.linspace(0, 1, 2 * num_points)) - mover.centroid
        )
        error_rs = np.abs(error_diff)
        error_thetas = np.angle(error_diff)

        u = mover.velocity.real - rs * mover.angular_velocity * np.sin(thetas)
        v = mover.velocity.imag + rs * mover.angular_velocity * np.cos(thetas)
        self.boundary_conditions[side] = None
        self.add_boundary_condition(side, f"u[{side}]", u)
        self.add_boundary_condition(side, f"v[{side}]", v)
        error_u = (
            mover.velocity.real
            - error_rs * mover.angular_velocity * np.sin(error_thetas)
        )
        error_v = (
            mover.velocity.imag
            + error_rs * mover.angular_velocity * np.cos(error_thetas)
        )
        self.error_values[side] = (error_u, error_v)

    def add_point(self, point):
        """Add an expression and value for a point to boundary conditions."""
        self.domain.add_point(point)

    def name_side(self, old, new):
        """Change the name of a side."""
        if self.boundary_conditions is not None:
            # TODO create an error class for this
            raise ValueError("Boundary conditions are already set.")
        self.domain._name_side(old, new)

    def group_sides(self, old_sides, new):
        """Group a list of sides together under a new name."""
        if self.boundary_conditions is not None:
            # TODO create an error class for this
            raise ValueError("Boundary conditions are already set.")
        self.domain._group_sides(old_sides, new)

    # BOUNDARY CONDITIONS
    def add_boundary_condition(self, side, expression, value):
        """Add an expression and value for a side to boundary conditions.

        The expression is stripped and added to the dictionary.
        """
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                side: None for side in self.domain.sides
            }
        expression = expression.strip().replace(" ", "")
        if side not in self.domain.sides:
            raise ValueError("side must be in domain.sides")
        if isinstance(self.boundary_conditions[side], Sequence):
            if len(self.boundary_conditions[side]) == 2:
                raise ValueError(
                    f"2 boundary conditions already set for side {side}"
                )
            if len(self.boundary_conditions[side]) == 1:
                self.boundary_conditions[side].append((expression, value))
        else:
            self.boundary_conditions[side] = [(expression, value)]

    def get_homogenous_problem(self):
        """Get the homogenous problem."""
        homogeneous_problem = self.copy()
        for side in homogeneous_problem.boundary_conditions.keys():
            side_bcs = homogeneous_problem.boundary_conditions[side]
            homog_bcs = [(expression, 0) for expression, value in side_bcs]
            homogeneous_problem.boundary_conditions[side] = homog_bcs
        return homogeneous_problem

    def check_boundary_conditions(self):
        """Check that the boundary conditions are valid."""
        for side in self.boundary_conditions.keys():
            if self.boundary_conditions[side] is None:
                raise ValueError(f"boundary condition not set for side {side}")
            if len(self.boundary_conditions[side]) != 2:
                raise ValueError(
                    f"2 boundary conditions not set for side {side}"
                )
            for expression, value in self.boundary_conditions[side]:
                self.validate(expression)
                if isinstance(value, str):
                    if not self.validate(value):
                        raise ValueError(
                            f"value {value} is not a valid expression."
                        )
                    continue
                if not isinstance(value, (int, float, np.ndarray)):
                    raise TypeError("value must be a numerical or string type")

    def validate(self, expression):
        """Check if the given expression has the correct syntax."""
        expression = expression.strip().replace(" ", "")
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        if expression.count("(") != expression.count(")"):
            raise ValueError("expression contains mismatched parentheses")
        # demand every dependent variable in the expression is followed by
        # (side)
        for dependent in DEPENDENT:
            while dependent in expression:
                index = expression.index(dependent)
                following = expression[index + len(dependent) :]
                if not following.startswith("["):
                    invalid = True
                    for side in self.domain.sides:
                        if side in expression and dependent in side:
                            # check for each occurance of side in expression
                            # whether dependent is in side
                            # TODO finish this logic
                            pass
                    if invalid:
                        raise ValueError(
                            f"dependent variable {dependent} not \
                            evaluated at a side"
                        )
                following = following[1:]
                closing = following.index("]")
                side = following[:closing]
                if side not in self.domain.sides:
                    raise ValueError(
                        f"trying to evaluate {dependent} at side {side} but \
                        {side} not in domain.sides"
                    )
                else:
                    expression = expression.replace(f"{dependent}[{side}]", "")

        for operation in OPERATIONS:
            expression = expression.replace(operation, "")
        if "[::-1]" in expression:
            print("[::-1] in expression")
        for quantity in INDEPENDENT:
            expression = expression.replace(quantity, "")
        # check decimals are surrounded by numbers
        while "." in expression:
            index = expression.index(".")
            if index == 0 or index == len(expression) - 1:
                raise ValueError(
                    f"expression contains invalid decimal: {expression}"
                )
            if (
                not expression[index - 1].isnumeric()
                or not expression[index + 1].isnumeric()
            ):
                raise ValueError(
                    f"expression contains invalid decimal: {expression}"
                )
            expression = expression.replace(".", "")
        for side in self.domain.sides:
            expression = expression.replace(side, "")
        for number in range(10):
            expression = expression.replace(str(number), "")
        # check if only numbers remain
        if expression != "":
            raise ValueError(
                f"expression contains invalid characters: {expression}"
            )
        return True

    def show(self):
        """Display the boundary points, poles and boundary conditions."""
        return NotImplemented

    def copy(self):
        """Return a copy of the current object."""
        new = copy.deepcopy(self)
        new.creation_time = perf_counter()
        return new
