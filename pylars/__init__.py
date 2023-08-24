"""Solve stokes flow problems using the lightning stokes method."""
from .numerics import va_orthogonalise
from .domain.domain import Domain
from .domain.periodic_domain import PeriodicDomain
from .domain.curved_domain import CurvedDomain
from .problem import Problem
from .solver import Solver
from .solution import Solution
from .analysis import Analysis
from .simulation_analysis import SimulationAnalysis
