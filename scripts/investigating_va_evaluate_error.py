"""Investigate relative error for va_evaluate against the MATLAB code."""
from pyls.numerics import (
    va_orthogonalise,
    va_evaluate,
)
from pyls import Domain
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# load the MATLAB results for lid driven cavity
n = 24
num_poles = 24
test_answers = loadmat(
    f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
)
Z_answer = test_answers["Z"]
poles_answer = test_answers["Pol"]
basis_answer = test_answers["R0"]
basis_deriv_answer = test_answers["R1"]
Q_answer = test_answers["Q"]
poles_answer = np.array([poles_answer[0, i] for i in range(4)]).reshape(
    4, num_poles
)

# setup the domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(
    corners,
    num_poles=num_poles,
    num_boundary_points=300,
    L=np.sqrt(2) * 1.5,
)
assert np.allclose(dom.boundary_points, Z_answer, atol=1e-15)
assert np.allclose(dom.poles, poles_answer, atol=1e-15)

hessenbergs, Q = va_orthogonalise(
    dom.boundary_points.reshape(1200, 1), n, poles=dom.poles
)
basis, basis_deriv = va_evaluate(
    dom.boundary_points, hessenbergs, poles=dom.poles
)

# investigate the relative error
column_error = np.linalg.norm(
    np.abs(basis - basis_answer) / np.abs(basis_answer), axis=0
)
column_deriv_error = np.linalg.norm(
    np.abs(basis_deriv - basis_deriv_answer) / np.abs(basis_deriv_answer),
    axis=0,
)
fig, ax = plt.subplots(1, 2)
ax[0].set_title("R0 vs MATLAB")
ax[0].set_xlabel("Column")
ax[0].set_ylabel("Relative Error")
ax[0].semilogy(column_error)
ax[1].set_title("R1 vs MATLAB")
ax[1].set_xlabel("Column")
ax[1].set_ylabel("Relative Error")
ax[1].semilogy(column_deriv_error)
plt.tight_layout()
plt.savefig("media/investigating_va_evaluate_error.pdf")
plt.show()
