"""The Solver class."""
from pylars.numerics import va_orthogonalise, va_evaluate, make_function
from pylars.problem import DEPENDENT, INDEPENDENT
from pylars.solution import Solution
import numpy as np
import scipy.linalg as linalg
from scipy.sparse import diags
import re
import pickle as pkl
import warnings


class Solver:
    """Solve lightning stokes problems on a given domain."""

    def __init__(self, problem, verbose=False):
        self.problem = problem
        self.verbose = verbose
        self.domain = problem.domain
        self.boundary_conditions = problem.boundary_conditions
        self.num_edge_points = self.domain.num_edge_points
        self.boundary_points = self.domain.boundary_points
        self.num_poles = self.domain.num_poles
        self.poles = self.domain.poles
        self.degree = self.domain.deg_poly

    def evaluate(self, expression, points):
        """Evaluate the given expression."""
        code_dependent = ["self.PSI", "self.U", "self.V", "self.P", "self.E12"]
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
                    self.evaluate(
                        expression1,
                        self.boundary_points[self.domain.indices[side]],
                    )
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
            self.domain.interior_laurents,
            self.domain.exterior_laurents,
        )
        # TODO check if e12 in boundary conditions before evaluating
        # second derivative
        (
            self.basis,
            self.basis_derivatives,
            self.basis_derivatives_2,
        ) = va_evaluate(
            self.boundary_points.reshape(-1, 1),
            self.hessenbergs,
            self.domain.poles,
            self.domain.interior_laurents,
            self.domain.exterior_laurents,
            second_deriv=True,
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
        if self.verbose:
            print("Generating Basis ...")
        self.hessenbergs, self.Q = va_orthogonalise(
            self.boundary_points,
            self.degree,
            self.domain.poles,
            self.domain.interior_laurents,
            self.domain.exterior_laurents,
        )
        (
            self.basis,
            self.basis_derivatives,
            self.basis_derivatives_2,
        ) = va_evaluate(
            self.boundary_points.reshape(-1, 1),
            self.hessenbergs,
            self.domain.poles,
            self.domain.interior_laurents,
            self.domain.exterior_laurents,
            second_deriv=True,
        )
        if self.verbose:
            print("Getting Dependents ...")
        self.get_dependents()
        if self.verbose:
            print("Constructing Linear System ...")
        self.construct_linear_system()
        if weight:
            self.weight_rows()
        if normalize:
            self.normalize()
        if self.verbose:
            print("A is of shape ", self.A.shape)
            print("Solving ...")
        if self.A.shape[0] < 4 * self.A.shape[1]:
            warnings.warn(
                "A is not tall skinny enough, answers may be unreliable.",
                Warning,
            )
        results = linalg.lstsq(self.A, self.b)
        self.coefficients = results[0]
        if results[1]:
            self.max_residual = np.sqrt(np.max(results[1]))
        else:
            if self.verbose:
                print("Evaluating Residual ...")
            self.max_residual = np.max(
                np.abs(self.A @ self.coefficients - self.b)
            )
        if self.verbose:
            print("Constructing Functions ...")
        self.functions = self.construct_functions()
        if pickle:
            self.pickle_solution(filename)
        return Solution(
            self.problem.copy(), *self.functions, self.max_residual
        )

    def construct_linear_system(self):
        """Use the basis functions to construct the linear system.

        Uses the boundary condition dictionary to construct the linear system
        with the basis functions from va_evaluate.
        """
        m = len(self.boundary_points)
        n = self.basis.shape[1]
        if not self.domain.interior_laurents:
            self.A1, self.A2 = np.zeros((m, 4 * n)), np.zeros((m, 4 * n))
            self.b1, self.b2 = np.zeros((m)), np.zeros((m))
        else:
            num_log = len(self.domain.interior_laurents)
            self.A1 = np.zeros((m, 4 * (n + num_log)))
            self.A2 = np.zeros((m, 4 * (n + num_log)))
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
        if self.domain.interior_laurents:
            return NotImplementedError
            n_row, n_col = self.A.shape
            self.b = np.vstack([self.b, np.zeros((4, 1))])
            self.A = np.vstack([self.A, np.zeros((4, n_col))])
            a = np.complex128(a)
            b = np.complex128(b)
            r0_a, r1_a = va_evaluate(
                a, self.hessenbergs, self.domain.poles, self.domain.laurents
            )
            centers = np.array(
                [laurent_series[0] for laurent_series in self.domain.laurents]
            ).reshape(1, -1)
            flog_a = np.log(a - centers) + (
                (a - centers) * np.log(a) + a
            ) / np.conj(a)
            glog_a = np.log(a - centers)
            flog_b = np.log(b - centers) + (
                (b - centers) * np.log(b) + b
            ) / np.conj(b)
            zero = np.zeros_like(r0_a, dtype=np.float64)
            zero_log = np.zeros_like(flog_a, dtype=np.float64)
            self.A[-4, :] = np.hstack(
                [
                    np.real(r0_a),
                    zero,
                    np.real(flog_a),
                    zero_log,
                    -np.imag(r0_a),
                    zero,
                    -np.imag(flog_a),
                    zero_log,
                ]
            )
            self.A[-3, :] = np.hstack(
                [
                    np.imag(r0_a),
                    zero,
                    np.imag(flog_a),
                    zero_log,
                    np.real(r0_a),
                    zero,
                    np.real(flog_a),
                    zero_log,
                ]
            )
            self.A[-2, :] = np.hstack(
                [
                    zero,
                    np.real(r0_a),
                    zero_log,
                    np.real(glog_a),
                    zero,
                    -np.imag(r0_a),
                    zero_log,
                    -np.imag(glog_a),
                ]
            )
            r0_b, r1_b = va_evaluate(
                b, self.hessenbergs, self.domain.poles, self.domain.laurents
            )
            self.A[-1, :] = np.hstack(
                [
                    np.real(r0_b),
                    zero,
                    np.real(flog_b),
                    zero_log,
                    -np.imag(r0_b),
                    zero,
                    np.real(flog_b),
                    zero_log,
                ]
            )
        else:
            n_row, n_col = self.A.shape
            self.b = np.vstack([self.b, np.zeros((4, 1))])
            self.A = np.vstack([self.A, np.zeros((4, n_col))])
            r0_a, r1_a = va_evaluate(a, self.hessenbergs, self.domain.poles)
            zero = np.zeros_like(r0_a, dtype=np.float64)
            self.A[-4, :] = np.hstack(
                [np.real(r0_a), zero, -np.imag(r0_a), zero]
            )
            self.A[-3, :] = np.hstack(
                [np.imag(r0_a), zero, np.real(r0_a), zero]
            )
            self.A[-2, :] = np.hstack(
                [zero, np.real(r0_a), zero, -np.imag(r0_a)]
            )
            r0_b, r1_b = va_evaluate(b, self.hessenbergs, self.domain.poles)
            self.A[-1, :] = np.hstack(
                [np.real(r0_b), zero, -np.imag(r0_b), zero]
            )

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
        basis = self.basis
        basis_deriv = self.basis_derivatives
        basis_deriv_2 = self.basis_derivatives_2
        conj_z = diags(z.conj().reshape(m))

        if not self.domain.interior_laurents:
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

            psi_1 = np.imag(conj_z @ basis)
            psi_2 = np.imag(basis)
            psi_3 = np.real(conj_z @ basis)
            psi_4 = np.real(basis)
            self.PSI = np.hstack((psi_1, psi_2, psi_3, psi_4))

            e12_1 = -np.imag(conj_z @ basis_deriv_2)
            e12_2 = -np.imag(basis_deriv_2)
            e12_3 = -np.real(conj_z @ basis_deriv_2)
            e12_4 = -np.real(basis_deriv_2)
            self.E12 = np.hstack((e12_1, e12_2, e12_3, e12_4))
        else:
            centers = np.array(
                [
                    laurent_series[0]
                    for laurent_series in self.domain.interior_laurents
                ]
            ).reshape(1, -1)
            num_log = centers.shape[1]
            z_m_centers = z - centers
            one_over_z = 1 / (z_m_centers)  # m x num_log array
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
            p_4 = np.zeros((m, num_log))
            p_5 = -np.imag(4 * basis_deriv)
            p_6 = np.zeros_like(p_1)
            p_7 = -np.imag(4 * one_over_z)
            p_8 = np.zeros((m, num_log))
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

            # TODO check e12_3 and e12_7 with Yidan.
            e12_1 = -np.imag(conj_z @ basis_deriv_2)
            e12_2 = -np.imag(basis_deriv_2)
            e12_3 = -np.imag(-conj_z * one_over_z**2 - one_over_z)
            e12_4 = -np.imag(-(one_over_z**2))
            e12_5 = -np.real(conj_z @ basis_deriv_2)
            e12_6 = -np.real(basis_deriv_2)
            e12_7 = -np.real(-conj_z * one_over_z**2 + one_over_z)
            e12_8 = -np.real(-(one_over_z**2))
            self.E12 = np.hstack(
                (e12_1, e12_2, e12_3, e12_4, e12_5, e12_6, e12_7, e12_8)
            )

    def pickle_solutions(self, filename):
        """Store the solution using pickle."""
        with open(filename, "wb") as f:
            pkl.dump(self.functions, f)

    def construct_functions(self):
        """Construct callable physical quantities from the coefficients."""
        # need to copy otherwise passing reference to domain object
        # which may change
        coefficients = self.coefficients.copy()
        hessenbergs = self.hessenbergs.copy()
        poles = self.domain.poles.copy()
        interior_laurents = self.domain.interior_laurents.copy()
        exterior_laurents = self.domain.exterior_laurents.copy()

        def psi(z):
            return make_function(
                "psi",
                z,
                coefficients,
                hessenbergs,
                poles,
                interior_laurents,
                exterior_laurents,
            )

        def uv(z):
            return make_function(
                "uv",
                z,
                coefficients,
                hessenbergs,
                poles,
                interior_laurents,
                exterior_laurents,
            )

        def p(z):
            return make_function(
                "p",
                z,
                coefficients,
                hessenbergs,
                poles,
                interior_laurents,
                exterior_laurents,
            )

        def omega(z):
            return make_function(
                "omega",
                z,
                coefficients,
                hessenbergs,
                poles,
                interior_laurents,
                exterior_laurents,
            )

        def eij(z):
            return make_function(
                "eij",
                z,
                coefficients,
                hessenbergs,
                poles,
                interior_laurents,
                exterior_laurents,
            )

        return psi, uv, p, omega, eij
