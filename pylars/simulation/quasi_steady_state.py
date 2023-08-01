"""Class for managing Quasi Steady State simulations."""
from pylars import Problem, Solver


class QuasiSteadySimulation:
    def __init__(self, initial_problem):
        if not isinstance(initial_problem, Problem):
            raise TypeError("initial_problem must be of type Problem.")
        self.initial_problem = initial_problem

    def add_update_rule(self, update_rule):
        """Add an update rule.

        Update rule should take in a Solution object from the
        previous timestep and return a Problem to be solved
        at the next timestep.
        """
        if not hasattr(update_rule, "__call__"):
            raise TypeError("update_rule must be a function.")
        self.update_rule = update_rule

    def run(self, n_steps, dt, filename=None):
        if filename is None:
            filename = self.initial_problem.name
        prob = self.initial_problem
        solver = Solver(self.initial_problem)
        solver.save()
        t = 0
        for i in range(1, n_steps):
            t += dt
            sol = solver.Solver(prob)
            sol.saveas(f"{filename}_{t:4f}.pkl")
            prob = self.update_rule(sol, t, dt)
