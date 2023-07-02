"""Solve stokes flow problems using the lighting stokes method.

Modules
-------
numerics:
    Contains functions implementing orthogonalisation.

domain:
    Contains the Domain class.
solver:
    Contains the Solver class.
analysis:
    Contains the Analysis class.
"""
from .numerics import va_orthogonalise  # noqa: F401
from .domain import Domain  # noqa: F401
from .solver import Solver  # noqa: F401
from .analysis import Analysis  # noqa: F401
