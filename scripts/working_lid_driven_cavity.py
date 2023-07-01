import numpy as np
import matplotlib.pyplot as plt
from pyls import Domain, Solver, Analysis
from pyls.numerics import va_orthogonalise, va_evaluate

# set up lid driven cavity problem
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
# dom.show()
sol = Solver(dom, degree=24)
# moving lid
sol.add_boundary_condition("0", "psi(0)", 0)
sol.add_boundary_condition("0", "u(0)", 1)
# wall boundary conditions
sol.add_boundary_condition("2", "psi(2)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("1", "psi(1)", 0)
sol.add_boundary_condition("1", "v(1)", 0)
sol.add_boundary_condition("3", "psi(3)", 0)
sol.add_boundary_condition("3", "v(3)", 0)
sol.hessenbergs, sol.Q = va_orthogonalise(
    sol.boundary_points, sol.degree, sol.domain.poles
)
sol.basis, sol.basis_derivatives = va_evaluate(
    sol.boundary_points, sol.hessenbergs, sol.domain.poles
)
sol.get_dependents()
sol.construct_linear_system()
sol.weight_rows()
sol.normalize()
sol.coefficients = np.linalg.lstsq(sol.A, sol.b, rcond=None)[0]
psi, uv, p, omega = sol.construct_functions()
# savemat("tests/data/lid_driven_cavity_matrix_python.mat", {"A": sol.A, "b": sol.b})


# plotting
x = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, x)
Z = X + 1j * Y
psi_100_100 = psi(Z.flatten()).reshape(100, 100)
uv_100_100 = uv(Z.flatten()).reshape(100, 100)
# plot the velocity magnitude
fig, ax = plt.subplots()
# interpolate using bilinear interpolation
speed = np.abs(uv_100_100)
ax.pcolormesh(X, Y, np.abs(uv_100_100), cmap="jet")
psi_min, psi_max = psi_100_100.min(), psi_100_100.max()
levels = np.linspace(psi_min, psi_max, 20)
ax.contour(X, Y, psi_100_100, colors="k", levels=20)
ax.set_aspect("equal")
psi_ratio = psi_min / psi_max
fac = np.max([psi_ratio, 1 / psi_ratio])
levels_moffat = np.linspace(1e-7, 5e-6, 10)
ax.contour(X, Y, psi_100_100, colors="y", levels=levels_moffat, linewidths=0.5)
plt.show()
