from pylars.numerics import aaa
import numpy as np
import cProfile as profile
import pstats

# profile the aaa function for approximating z conjugate on an ellipse
# with 1000 points


def run(a, b):
    """Profile the aaa function."""
    t = np.linspace(0, 1, 1000)
    z = a * np.cos(2 * np.pi * t) + b * 1j * np.sin(2 * np.pi * t)
    zc = np.conj(z)
    aaa(zc, z, mmax=100)
    return None


if __name__ == "__main__":
    with profile.Profile() as pr:
        run(1, 3)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
    stats.dump_stats("./scripts/benchmarking/aaa.prof")
