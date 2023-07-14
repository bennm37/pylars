"""The Solver class."""
from pylars.numerics import va_orthogonalise, va_evaluate, make_function
from pylars.problem import DEPENDENT, INDEPENDENT
from pylars.solution import Solution
import numpy as np
import scipy.linalg as linalg
from scipy.sparse import diags
import re
import pickle as pkl


class Solver:
    """Solve lightning stokes problems on a given domain."""

    def __init__(self, problem, least_squares="iterative"):
        self.problem = problem
        self.domain = problem.domain
        self.boundary_conditions = problem.boundary_conditions
        self.least_squares = least_squares
        self.num_edge_points = self.domain.num_edge_points
        self.boundary_points = self.domain.boundary_points
        self.num_poles = self.domain.num_poles
        self.poles = self.domain.poles
        self.degree = self.domain.deg_poly

    def evaluate(self, expression, points):
        """Evaluate the given expression."""
        # TODO add flip to evaluate
        code_dependent = [
            "self.PSI",
            "self.U",
            "self.V",
            "self.P",
        ]
        for identifier, code in zip(DEPENDENT, code_dependent):
            identifier += "["
            code += "["
            expression = expression.replace(identifier, code)
        for side in self.domain.sides:
            expression = re.sub(
                f"\[{side}\]", f'[self.domain.indices["{side}"]]', expression
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

    def setup(self):
        """Get basis functions and derivatives and dependent variables."""
        self.hessenbergs, self.Q = va_orthogonalise(
            self.boundary_points.reshape(-1, 1),
            self.degree,
            self.domain.poles,
            self.domain.laurents,
        )
        self.basis, self.basis_derivatives = va_evaluate(
            self.boundary_points.reshape(-1, 1),
            self.hessenbergs,
            self.domain.poles,
            self.domain.laurents,
        )
        self.get_dependents()

    def solve(
        self,
        pickle=False,
        filename="solution.pickle",
        check=True,
        normalize=True,
        weight=True,
    ):
        """Set up the solver and solve the least squares problem.

        Reutrns the functions as a list of functions.
        """
        if check:
            self.problem.check_boundary_conditions()
        self.hessenbergs, self.Q = va_orthogonalise(
            self.boundary_points,
            self.degree,
            self.domain.poles,
            self.domain.laurents,
        )
        self.basis, self.basis_derivatives = va_evaluate(
            self.boundary_points,
            self.hessenbergs,
            self.domain.poles,
            self.domain.laurents,
        )
        self.get_dependents()
        self.construct_linear_system()
        if weight:
            self.weight_rows()
        if normalize:
            self.normalize()
        self.coefficients = linalg.lstsq(self.A, self.b)[0]
        self.functions = self.construct_functions()
        if pickle:
            self.pickle_solution(filename)
        return Solution(*self.functions)

    def construct_linear_system(self):
        """Use the basis functions to construct the linear system.

        Uses the boundary condition dictionary to construct the linear system
        with the basis functions from va_evaluate.
        """
        m = len(self.boundary_points)
        n = self.basis.shape[1]
        if not self.domain.laurents:
            self.A1, self.A2 = np.zeros((m, 4 * n)), np.zeros((m, 4 * n))
            self.b1, self.b2 = np.zeros((m)), np.zeros((m))
        else:
            num_laurent = len(self.domain.laurents)
            self.A1 = np.zeros((m, 4 * (n + num_laurent)))
            self.A2 = np.zeros((m, 4 * (n + num_laurent)))
            self.b1, self.b2 = np.zeros((m)), np.zeros((m))
        self.apply_boundary_conditions()
        self.A = np.vstack((self.A1, self.A2))
        self.b = np.vstack((self.b1.reshape(m, 1), self.b2.reshape(m, 1)))

    def weight_rows(self):
        """Weight the rows by the distance to the nearest corner."""
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

    def normalize(self, a=0, b=1):
        """Normalize f and g so that they are unique."""
        n_row, n_col = self.A.shape
        self.b = np.vstack([self.b, np.zeros((4, 1))])
        self.A = np.vstack([self.A, np.zeros((4, n_col))])
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
        # NOTE for laurent series the coefficient vector is
        # structured (f_r, g_r, f_lr, g_lr, f_i, g_i, f_li, g_li)
        # where terms with an l correspond to the log terms.
        # TODO verbose but I guess clear. Could be more concise?
        # TODO is copying large arrays like this expensive?
        z = self.boundary_points
        m = len(z)
        basis, basis_deriv = self.basis, self.basis_derivatives
        conj_z = diags(z.conj().reshape(m))

        if not self.domain.laurents:
            u_1 = np.real(conj_z @ basis_deriv - basis)
            u_2 = np.real(basis_deriv)
            u_3 = -np.imag(conj_z @ basis_deriv - basis)
            u_4 = -np.imag(basis_deriv)
            self.U = np.hstack((u_1, u_2, u_3, u_4))

            v_1 = -np.imag(conj_z @ basis_deriv + basis)
            v_2 = -np.imag(basis_deriv)
            v_3 = -np.real(conj_z @ basis_deriv + basis)
            v_4 = -np.real(basis_deriv)
            self.V = np.hstack((v_1, v_2, v_3, v_4))

            p_1 = np.real(4 * basis_deriv)
            p_2 = np.zeros_like(p_1)
            p_3 = -np.imag(4 * basis_deriv)
            p_4 = np.zeros_like(p_1)
            self.P = np.hstack((p_1, p_2, p_3, p_4))

            s_1 = np.imag(conj_z @ basis)
            s_2 = np.imag(basis)
            s_3 = np.real(conj_z @ basis)
            s_4 = np.real(basis)
            self.PSI = np.hstack((s_1, s_2, s_3, s_4))
        else:
            centers = np.array(
                [laurent_series[0] for laurent_series in self.domain.laurents]
            ).reshape(1, -1)
            num_laurent = centers.shape[1]
            z_m_centers = z - centers
            one_over_z = 1 / (z_m_centers)  # m x num_laurent array
            log_z = np.log(z_m_centers)
            u_1 = np.real(conj_z @ basis_deriv - basis)
            u_2 = np.real(basis_deriv)
            u_3 = np.real(conj_z @ one_over_z - 2 * log_z)
            u_4 = np.real(one_over_z)
            u_5 = -np.imag(conj_z @ basis_deriv - basis)
            u_6 = -np.imag(basis_deriv)
            u_7 = -np.imag(conj_z * one_over_z)
            u_8 = -np.imag(one_over_z)
            self.U = np.hstack((u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8))

            v_1 = -np.imag(conj_z @ basis_deriv + basis)
            v_2 = -np.imag(basis_deriv)
            v_3 = np.imag(-conj_z * one_over_z)
            v_4 = np.imag(-one_over_z)
            v_5 = -np.real(conj_z @ basis_deriv + basis)
            v_6 = -np.real(basis_deriv)
            v_7 = np.real(-conj_z * one_over_z - 2 * log_z)
            v_8 = np.real(-one_over_z)
            self.V = np.hstack((v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8))

            p_1 = np.real(4 * basis_deriv)
            p_2 = np.zeros_like(p_1)
            p_3 = np.real(4 * one_over_z)
            p_4 = np.zeros((m, num_laurent))
            p_5 = -np.imag(4 * basis_deriv)
            p_6 = np.zeros_like(p_1)
            p_7 = -np.imag(4 * one_over_z)
            p_8 = np.zeros((m, num_laurent))
            self.P = np.hstack((p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8))

            psi_1 = np.imag(conj_z @ basis)
            psi_2 = np.imag(basis)
            psi_3 = np.imag(conj_z * log_z - z_m_centers * log_z + z)
            psi_4 = np.imag(log_z)
            psi_5 = np.real(conj_z @ basis)
            psi_6 = np.real(basis)
            psi_7 = np.real(conj_z * log_z + z_m_centers * log_z - z)
            psi_8 = np.real(log_z)
            self.PSI = np.hstack(
                (psi_1, psi_2, psi_3, psi_4, psi_5, psi_6, psi_7, psi_8)
            )

    def pickle_solutions(self, filename):
        """Store the solution using pickle."""
        with open(filename, "wb") as f:
            pkl.dump(self.functions, f)

    def construct_functions(self):
        """Construct callable physical quantities from the coefficients."""

        def psi(z):
            return make_function(
                "psi",
                z,
                self.coefficients,
                self.hessenbergs,
                self.domain.poles,
                self.domain.laurents,
            )

        def uv(z):
            return make_function(
                "uv",
                z,
                self.coefficients,
                self.hessenbergs,
                self.domain.poles,
                self.domain.laurents,
            )

        def p(z):
            return make_function(
                "p",
                z,
                self.coefficients,
                self.hessenbergs,
                self.domain.poles,
                self.domain.laurents,
            )

        def omega(z):
            return make_function(
                "omega",
                z,
                self.coefficients,
                self.hessenbergs,
                self.domain.poles,
                self.domain.laurents,
            )

        return psi, uv, p, omega
