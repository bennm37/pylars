import cProfile
import pstats
from pyls.numerics import va_orthogonalise, va_orthogonalise_jit
import numpy as np


def profile():
    n_points = 12000
    degree = 100
    Z = np.exp(1j * np.linspace(0, 2 * np.pi, n_points)).reshape(n_points, 1)
    hessenbergs, Q = va_orthogonalise(Z, degree)
    hessenbergs_jit, Q_jit = va_orthogonalise_jit(Z, degree)

    with cProfile.Profile() as pr:
        va_orthogonalise(Z, degree)
    with cProfile.Profile() as pr_jit:
        va_orthogonalise_jit(Z, degree)

    stats = pstats.Stats(pr)
    stats_jit = pstats.Stats(pr_jit)
    # sort by cumulative time
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats_jit.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(5)
    stats_jit.print_stats(5)


if __name__ == "__main__":
    profile()
