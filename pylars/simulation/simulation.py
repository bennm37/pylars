"""Base Class for simulations."""
from tqdm import tqdm
import numpy as np


class Simulation:
    """Abstract Simulation Class for Quasi Steady State Simulations."""

    def __init__(self, base_problem):
        self.base_problem = base_problem
        self.results = {}
        self.name = "AbstractSimulation"

    def run(self, start, n_steps, dt, name=None):
        """Run the simulation."""
        end = start + n_steps * dt
        self.start, self.end, self.dt = start, end, dt
        if name is None:
            name = self.name
        self.times = np.linspace(start, start + (n_steps - 1) * dt, n_steps)
        self.n_steps = n_steps
        # create a progress bar for loop
        for k, time in enumerate(tqdm(self.times, desc=self.name)):
            self.update(k, dt)
        self.results["time_data"] = self.times
        return self.results

    def update(self):
        """Abstract Update Method."""
        return NotImplemented
