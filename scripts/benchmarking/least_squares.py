from pylars import Problem, Solver
from pylars.domain.generation import generate_rv_circles
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import time


def generate_linear_system(seed, bound=1):
    np.random.seed(seed)
    prob = Problem()
    length = 2 * bound
    corners = [
        -1 * bound - 1j * bound,
        1 * bound - 1j * bound,
        1 * bound + 1j * bound,
        -1 * bound + 1j * bound,
    ]
    prob.add_periodic_domain(
        length,
        length,
        num_edge_points=600,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )
    porosity = 0.95
    rv = lognorm.rvs
    rv_args = {"s": 0.5, "scale": 0.27, "loc": 0.0}
    centroids, radii = generate_rv_circles(
        porosity=porosity,
        rv=rv,
        rv_args=rv_args,
        length=length,
        min_dist=0.05,
    )
    n_circles = len(centroids)
    print(f"{n_circles = }")
    for centroid, radius in zip(centroids, radii):
        prob.add_periodic_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=300,
            deg_laurent=50,
            centroid=centroid,
            mirror_laurents=True,
            image_laurents=True,
        )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
    prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    interiors = [str(i) for i in range(4, 4 + n_circles)]
    for interior in interiors:
        prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
        prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)

    solver = Solver(prob, verbose=True)
    solver.setup()
    solver.construct_linear_system()
    A = solver.A
    b = solver.b
    print(A.shape)
    return A, b


def test_solvers():
    []


if __name__ == "__main__":
    num_tests = 10
    n_solvers = 4
    bound = 1
    times = np.zeros((num_tests, n_solvers))
    residuals = np.zeros((num_tests, n_solvers))
    for i in range(num_tests):
        A, b = generate_linear_system(i, bound)
        start_torch = time.perf_counter()
        # if torch.cuda.is_available():
        #     dev = "cuda:0"
        # else:
        #     dev = "cpu"
        dev = "cpu"
        device = torch.device(dev)
        A_torch = torch.tensor(A)
        b_torch = torch.tensor(b)
        A_torch = A_torch.to(device)
        b_torch = b_torch.to(device)
        sol_torch = torch.linalg.lstsq(
            A_torch, b_torch, rcond=None, driver="gelsd"
        )
        end_torch = time.perf_counter()
        time_torch = end_torch - start_torch
        residual_torch = torch.max(
            torch.abs(torch.matmul(A_torch, sol_torch[0]) - b_torch)
        )
        print(f"Residual: {residual_torch}")
        print("Finished torch.")
        start_np = time.perf_counter()
        sol_np = np.linalg.lstsq(A, b, rcond=None)
        end_np = time.perf_counter()
        time_np = end_np - start_np
        residual_np = np.max(np.abs(A @ sol_np[0] - b))
        print("Finished numpy.")
        start_scipy = time.perf_counter()
        sol_scipy = scipy.linalg.lstsq(A, b, lapack_driver="gelsd")
        end_scipy = time.perf_counter()
        time_scipy = end_scipy - start_scipy
        residual_scipy = np.max(np.abs(A @ sol_scipy[0] - b))
        print("Finished scipy.")
        print("Finished cupy.")
        times[i, 0] = time_torch
        times[i, 1] = time_np
        times[i, 2] = time_scipy
        residuals[i, 0] = residual_torch
        residuals[i, 1] = residual_np
        residuals[i, 2] = residual_scipy
        print(f"{times = }")
        print(f"{residuals = }")
    times_df = pd.DataFrame(times, columns=["torch", "numpy", "scipy", "cupy"])
    times_df.boxplot()
    plt.show()
