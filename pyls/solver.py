from pyls.numerics import get_hessenbergs, get_basis_functions
import scipy.linalg as linalg
import pickle as pkl


class Solver:
    def __init__(self, domain, degree):
        self.domain = domain
        self.check_input()
        self.num_boundary_points = self.domain.num_boundary_points
        self.boundary_points = self.domain.boundary_points
        self.num_poles = self.domain.num_poles
        self.domain.poles = self.domain.poles
        self.degree = degree
        self.boundary_conditions = dict()

    def add_boundary_condition(self, boundary_condition):
        """Take a dictionary of boundary conditions and
        store them to the solver."""
        pass

    def check_input(self):
        pass

    def check_boundary_conditions(self):
        pass

    def solve(self, pickle=False, filename="solution.pickle"):
        self.check_boundary_conditions()
        self.hessenbergs = get_hessenbergs(
            self.boundary_points, self.degree, self.poles
        )
        self.basis_functions = get_basis_functions(
            self.boundary_points, self.hessenbergs
        )
        self.construct_linear_system()
        results = linalg.lstsq(self.A, self.b)
        self.coefficients = results[0]
        self.residuals = results[1]
        self.functions = self.construct_functions()
        if pickle:
            self.pickle_solution(filename)
        return self.functions

    def construct_linear_system(self, basis_functions):
        pass

    def pickle_solutions(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self.functions, f)

    def construct_functions(self, basis_functions):
        pass
