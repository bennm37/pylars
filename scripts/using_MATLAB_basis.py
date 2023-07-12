from pylars import Domain, Solver
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import lstsq

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_edge_points=300, num_poles=24)
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "psi(0)", 0)
sol.add_boundary_condition("0", "u(0)", 1)
sol.add_boundary_condition("2", "psi(2)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("1", "psi(1)", 1)
sol.add_boundary_condition("1", "psi(1)", 1)
sol.add_boundary_condition("3", "psi(3)", 3)
sol.add_boundary_condition("3", "psi(3)", 3)
n = 24
num_poles = 24
test_answers = loadmat(
    f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
)
Z_answer = test_answers["Z"]
basis = test_answers["R0"]
basis_deriv = test_answers["R1"]
hessenbergs = test_answers["Hes"]
hessenbergs = [hessenbergs[0, i] for i in range(5)]
sol.basis = basis
sol.basis_derivatives = basis_deriv
sol.hessenbergs = hessenbergs
sol.get_dependents()
assert np.allclose(sol.U, test_answers["U"])
assert np.allclose(sol.V, test_answers["V"])
assert np.allclose(sol.PSI, test_answers["PSI"])
sol.construct_linear_system()
sol.coefficients = lstsq(sol.A, sol.b)[0]
psi, uv, p, omega = sol.construct_functions()
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")


x = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, x)
Z = X + 1j * Y
psi_100_100 = psi(Z.flatten()).reshape(100, 100)
uv_100_100 = uv(Z.flatten()).reshape(100, 100)
# plot the velocity magnitude
fig, ax = plt.subplots()
ax.pcolor(X, Y, np.abs(uv_100_100), cmap="jet")
ax.contour(X, Y, psi_100_100, colors="k", levels=20)
ax.set_aspect("equal")
plt.show()
