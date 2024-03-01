import os
import sys
import cProfile as profile
import pstats
import shutil

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../example_flow")),
)
from log_normal_circles import run

if __name__ == "__main__":
    shutil.rmtree("./data/profiling")
    parameters = {
        "project_name": "profiling",
        "porosity": 0.95,
        "n_max": 1,
        "alpha": 0.05,
        "eps_CLT": 1.0,
        "rv": "lognorm",
        "rv_args": {"s": 0.5, "scale": 0.275, "loc": 0.0},
        "lengths": [15],
        "p_drop": 100,
    }
    with profile.Profile() as pr:
        run(parameters)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
    stats.dump_stats("./scripts/benchmarking/log_normal.prof")
