"""Base Class for simulations."""
from tqdm import tqdm
import numpy as np


class Simulation:
    """Abstract Simulation Class for Quasi Steady State Simulations."""

    def __init__(self, base_problem):
        self.base_problem = base_problem
        self.results = {}
        self.name = "AbstractSimulation"

    def run(self, start, end, dt, name=None):
        """Run the simulation."""
        self.start, self.end, self.dt = start, end, dt
        if name is None:
            name = self.name
        self.times = np.arange(start, end, dt)
        self.n_steps = self.times.shape[0]
        # create a progress bar for loop
        for k, time in enumerate(tqdm(self.times, desc=self.name)):
            self.update(k, dt)
        return self.results

    def update(self):
        """Abstract Update Method."""
        return NotImplemented
