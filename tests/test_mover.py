"""Test the Mover class from pylars.simulation."""


def test_intialise_mover():
    """Test that the Mover class can be initialised."""
    from pylars.simulation import Mover
    import numpy as np

    centroid = 0.0 + 0.0j
    R = 0.3
    curve = lambda t: R * np.exp(2j * np.pi * t)
    curve_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)
    rho = 1
    cell = Mover(
        curve=curve, deriv=curve_deriv, centroid=centroid, density=rho
    )
    assert cell is not None
    # assert cell.mass == rho * R**2 * np.pi
    # assert cell.moi == rho * (R) ** 4 * np.pi / 2
    assert isinstance(cell.angle, float)
    assert isinstance(cell.angular_velocity, float)
    assert isinstance(cell.velocity, float) or isinstance(
        cell.velocity, complex
    )
    assert isinstance(cell.centroid, float) or isinstance(
        cell.centroid, complex
    )


if __name__ == "__main__":
    test_intialise_mover()
