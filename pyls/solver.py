from pyls.numerics import va_orthogonalise, va_evaluate, make_function
from collections.abc import Sequence
import scipy.linalg as linalg
import numpy as np
from scipy.sparse import diags
import re
import pickle as pkl

OPERATIONS = ["[::-1]", "+", "-", "*", "/", "**", "(", ")"]
DEPENDENT = ["psi", "u", "v", "p"]
INDEPENDENT = ["x", "y"]


class Solver:
    """Solve lighting stokes problems on a given domain."""

    def __init__(self, domain, degree):
        self.domain = domain
        self.check_input()
        self.num_boundary_points = self.domain.num_boundary_points
        self.boundary_points = self.domain.boundary_points
        self.num_poles = self.domain.num_poles
        self.domain.poles = self.domain.poles
        self.degree = degree
        self.boundary_conditions = {side: None for side in self.domain.sides}

    def add_boundary_condition(self, side, expression, value):
        """Add an expression and value for a side to boundary conditions.

        The expression is stripped and added to the dictionary.
        """
        expression = expression.strip().replace(" ", "")
        self.validate(expression)
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
                if not following.startswith("("):
                    raise ValueError(
                        f"dependent variable {dependent} not evaluated at a \
                        side"
                    )
                following = following[1:]
                closing = following.index(")")
                side = following[:closing]
                if side not in self.domain.sides:
                    raise ValueError(
                        f"trying to evaluate {dependent} at side {side} but \
                        {side} not in domain.sides"
                    )
                else:
                    expression = expression.replace(f"{dependent}({side})", "")

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

    def evaluate(self, expression, points):
        """Evaluate the given expression."""
        # TODO add flip to evaluate
        code_dependent = [
            "self.stream_function",
            "self.U",
            "self.V",
            "self.pressure",
        ]
        for identifier, code in zip(DEPENDENT, code_dependent):
            identifier += "("
            code += "("
            expression = expression.replace(identifier, code)
        for side in self.domain.sides:
            expression = re.sub(
                f"\({side}\)", f'[self.domain.indices["{side}"]]', expression
            )
        for identifier, code in zip(
            INDEPENDENT,
            ["np.real(points)", "np.imag(points)"],
        ):
            expression = expression.replace(identifier, code)
        result = eval(expression)
        return result

    def apply_boundary_conditions(self):
        """Apply the boundary conditions to the linear system."""
        for side in self.boundary_conditions.keys():
            try:
                try:
                    expression1, value1 = self.boundary_conditions[side][0]
                    self.A1[self.domain.indices[side]] = self.evaluate(
                        expression1,
                        self.boundary_points[self.domain.indices[side]],
                    )
                    if isinstance(value1, str):
                        self.b1[self.domain.indices[side]] = self.evaluate(
                            value1,
                            self.boundary_points[self.domain.indices[side]],
                        ).reshape(-1)
                    else:
                        self.b1[self.domain.indices[side]] = value1
                    expression2, value2 = self.boundary_conditions[side][1]
                    self.A2[self.domain.indices[side]] = self.evaluate(
                        expression2,
                        self.boundary_points[self.domain.indices[side]],
                    )
                    if isinstance(value2, str):
                        self.b2[self.domain.indices[side]] = self.evaluate(
                            value2,
                            self.boundary_points[self.domain.indices[side]],
                        ).reshape(-1)
                    else:
                        self.b2[self.domain.indices[side]] = value2
                except IndexError:
                    print("Only One Boundary Condition set for side", side)
            except TypeError:
                print("Boundary Condition not set for side", side)

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

    def check_input(self):
        pass

    def setup(self):
        """Get basis functions and derivatives and dependent variables."""
        self.hessenbergs, self.Q = va_orthogonalise(
            self.boundary_points.reshape(-1, 1), self.degree, self.domain.poles
        )
        self.basis, self.basis_derivatives = va_evaluate(
            self.boundary_points.reshape(-1, 1),
            self.hessenbergs,
            self.domain.poles,
        )
        self.get_dependents()

    def solve(self, pickle=False, filename="solution.pickle", check=True):
        """Setup the solver and solve the least squares problem.

        Reutrns the functions as a list of functions."""
        if check:
            self.check_boundary_conditions()
        self.hessenbergs, self.Q = va_orthogonalise(
            self.boundary_points, self.degree, self.domain.poles
        )
        self.basis, self.basis_derivatives = va_evaluate(
            self.boundary_points, self.hessenbergs, self.domain.poles
        )
        self.get_dependents()
        self.construct_linear_system()
        self.weight_rows()
        self.normalize()
        self.results = linalg.lstsq(self.A, self.b)
        self.coefficients = self.results[0]
        self.residuals = self.results[1]
        self.functions = self.construct_functions()
        if pickle:
            self.pickle_solution(filename)
        return self.functions

    def construct_linear_system(self):
        # TODO this could move into solve for conciseness
        """Use the basis functions to construct the linear system.

        Uses the boundary condition dictionary to construct the linear system
        with the basis functions from va_evaluate.
        """
        m = len(self.boundary_points)
        n = self.basis.shape[1]
        self.A1, self.A2 = np.zeros((m, 4 * n)), np.zeros((m, 4 * n))
        self.b1, self.b2 = np.zeros((m)), np.zeros((m))
        self.apply_boundary_conditions()
        self.A = np.vstack((self.A1, self.A2))
        self.b = np.vstack((self.b1.reshape(m, 1), self.b2.reshape(m, 1)))

    def weight_rows(self):
        # weight the rows by the distance to the nearest corner
        m = len(self.boundary_points)
        row_weights = np.min(
            np.abs(
                self.boundary_points.reshape(-1)[:, np.newaxis]
                - self.domain.corners[np.newaxis, :]
            ),
            axis=1,
        ).reshape(m, 1)
        row_weights = np.vstack([row_weights, row_weights])
        sparse_row_weights = diags(row_weights.reshape(-1))
        self.A = sparse_row_weights @ self.A
        self.b = sparse_row_weights @ self.b
        return row_weights

    def normalize(self, a=0, b=0):
        """Normalize f and g so that they are unique."""
        h, l = self.A.shape
        self.b = np.vstack([self.b, np.zeros((4, 1))])
        self.A = np.vstack([self.A, np.zeros((4, l))])
        r0_a, r1_a = va_evaluate(a, self.hessenbergs, self.domain.poles)
        zero = np.zeros_like(r0_a, dtype=np.float64)
        self.A[-4, :] = np.hstack([np.real(r0_a), zero, -np.imag(r0_a), zero])
        self.A[-3, :] = np.hstack([np.imag(r0_a), zero, np.real(r0_a), zero])
        self.A[-2, :] = np.hstack([zero, np.real(r0_a), zero, -np.imag(r0_a)])
        r0_b, r1_b = va_evaluate(b, self.hessenbergs, self.domain.poles)
        self.A[-1, :] = np.hstack([np.real(r0_b), zero, -np.imag(r0_b), zero])

    def get_dependents(self):
        """Create the dependent variable arrays.

        Uses the basis to construct the physical quantities needed to
        set boundary conditions.
        """
        # NOTE paper notation and MATLAB code are different.
        # In paper x is stacked (f_r,f_i, g_r, g_i) but in MATLAB
        # stacked (f_r, g_r, f_i, g_i). This is convenient as then
        # splitting the coefficient vector into real and imaginary
        # parts is slightly easier. Changed to MATLAB notation.
        # TODO verbose but I guess clear. Could be more concise?
        # TODO is copying large arrays like this expensive?
        Z = self.boundary_points
        m = len(Z)
        basis, basis_deriv = self.basis, self.basis_derivatives
        z_conj = diags(Z.conj().reshape(m))

        u_1 = np.real(z_conj @ basis_deriv - basis)
        u_2 = np.real(basis_deriv)
        u_3 = -np.imag(z_conj @ basis_deriv - basis)
        u_4 = -np.imag(basis_deriv)
        self.U = np.hstack((u_1, u_2, u_3, u_4))

        v_1 = -np.imag(z_conj @ basis_deriv + basis)
        v_2 = -np.imag(basis_deriv)
        v_3 = -np.real(z_conj @ basis_deriv + basis)
        v_4 = -np.real(basis_deriv)
        self.V = np.hstack((v_1, v_2, v_3, v_4))

        p_1 = np.real(4 * basis_deriv)
        p_2 = np.zeros_like(p_1)
        p_3 = -np.imag(4 * basis_deriv)
        p_4 = np.zeros_like(p_1)
        self.pressure = np.hstack((p_1, p_2, p_3, p_4))

        s_1 = np.imag(z_conj @ basis)
        s_2 = np.imag(basis)
        s_3 = np.real(z_conj @ basis)
        s_4 = np.real(basis)
        self.stream_function = np.hstack((s_1, s_2, s_3, s_4))

    def pickle_solutions(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self.functions, f)

    def construct_functions(self):
        def psi(z):
            return make_function(
                "psi",
                z,
                self.coefficients,
                self.hessenbergs,
                self.domain.poles,
            )

        def uv(z):
            return make_function(
                "uv", z, self.coefficients, self.hessenbergs, self.domain.poles
            )

        def p(z):
            return make_function(
                "p", z, self.coefficients, self.hessenbergs, self.domain.poles
            )

        def omega(z):
            return make_function(
                "omega",
                z,
                self.coefficients,
                self.hessenbergs,
                self.domain.poles,
            )

        return psi, uv, p, omega
