"""Common values and functions to be used throughout the tests."""
ATOL, RTOL = 1e-12, 1e-6


def load_lid_driven_cavity(n, num_poles):
    """Load the lid driven cavity data."""
    from scipy.io import loadmat

    return loadmat(f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat")
