"""Solve Poiseiulle flow with stream function boundary conditions."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt


def check_flow(sol):
    """Checks if the flow is doubly periodic."""
    dom = sol.domain
    psi, uv, p, omega = sol.functions
    boundary_top = dom.boundary_points[dom.indices["0"]]
    boundary_bottom = dom.boundary_points[dom.indices["2"]]
    boundary_left = dom.boundary_points[dom.indices["1"]]
    boundary_right = dom.boundary_points[dom.indices["3"]]
    assert np.allclose(uv(boundary_top), uv(boundary_bottom[::-1]))
    assert np.allclose(uv(boundary_left), uv(boundary_right[::-1]))
    print("Flow is doubly periodic")
    #  print psi, u and v at the corners
    print("U at corners: ", uv(corners).real)
    print("V at corners: ", uv(corners).imag)
    print("Psi at corners: ", psi(corners))
    left_flux = np.mean(uv(boundary_left)) * 2
    right_flux = np.mean(uv(boundary_right)) * 2
    top_flux = np.mean(uv(boundary_top)) * 2
    bottom_flux = np.mean(uv(boundary_bottom)) * 2
    print(f"Left flux is {left_flux}")
    print(f"Right flux is {right_flux}")
    print(f"Top flux is {top_flux}")
    print(f"Bottom flux is {bottom_flux}")


def animate_circle(sol_1, sol_2):
    """Animate a linear combination of the solutions."""
    theta = np.linspace(0, np.pi, 100)
    a_values = np.cos(theta)
    b_values = np.sin(theta)
    an = Analysis(dom, sol_left)
    fig, ax, anim = an.animate_combination(
        sol_top, a_values, b_values, gapa=1, gapb=0, n_tile=3
    )
    anim.save("media/linear_1100.mp4", fps=20)


# solve the tb and lr problems
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0, spacing="linear")
# dom.show()
sol_left = Solver(dom, 24)
a1, a2, a3, a4 = 1, 0, 0, 0
sol_left.add_boundary_condition("0", "u(0)", 0)
sol_left.add_boundary_condition("0", "psi(0)", a1)
sol_left.add_boundary_condition("2", "u(2)", 0)
sol_left.add_boundary_condition("2", "psi(2)", a2)
sol_left.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol_left.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
psi_left, uv_left, p_left, omega_left = sol_left.solve(
    check=False, weight=False
)
sol_top = Solver(dom, 24)
sol_top.add_boundary_condition("1", "v(1)", 0)
sol_top.add_boundary_condition("1", "psi(1)", f"(y - 1)**2 * (y + 1)**2+{a3}")
# sol_top.add_boundary_condition("1", "psi(1)", a3)
sol_top.add_boundary_condition("3", "v(3)", 0)
sol_top.add_boundary_condition("3", "psi(3)", f"(y - 1)**2 * (y + 1)**2+{a4}")
# sol_top.add_boundary_condition("3", "psi(3)", a4)
sol_top.add_boundary_condition("0", "u(0)-u(2)[::-1]", 0)
sol_top.add_boundary_condition("0", "v(0)-v(2)[::-1]", 0)
psi_top, uv_top, p_top, omega_top = sol_top.solve(check=False, weight=False)

# combine the solutions
a, b = 1, 1
sol_combined = a * sol_left + b * sol_top
check_flow(sol_combined)
# animate_circle(sol_left, sol_top)

# ANIMATE CHANGIN a AND b
# theta = np.linspace(0, 1, 100)
# a_values = np.ones_like(theta)
# b_values = theta
# an = Analysis(dom, sol_left)
# fig, ax, anim = an.animate_combination(
#     sol_top, a_values, b_values, gapa=1, gapb=0, n_tile=3
# )
# anim.save("media/linear_1100.mp4", fps=20)

# SAVE SNAPSHOTS
# a_snap = [1.0, 1.0, 1.0]
# b_snap = [0.0, 0.5, 1.0]
# for a, b in zip(a_snap, b_snap):
#     sol_combined = a * sol_left + b * sol_top
#     an = Analysis(dom, sol_combined)
#     fig, ax = an.plot_periodic(a, b, gapa=1, gapb=0, n_tile=3)
#     plt.tight_layout()
#     plt.savefig(f"media/linear_{a:.1f}_{b:.1f}.pdf")
