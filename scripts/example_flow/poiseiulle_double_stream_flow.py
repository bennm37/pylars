"""Solve Poiseiulle flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt


def check_flow(problem, solution):
    """Checks if the flow is doubly periodic."""
    dom = problem.domain
    psi, uv, p, omega = solution.functions
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


# solve the tb and lr problems
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob_left = Problem()
prob_left.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=0, deg_poly=24, spacing="linear"
)
a1, a2, a3, a4 = 1, 0, 0, 0
prob_left.add_boundary_condition("0", "u[0]", 0)
prob_left.add_boundary_condition("0", "psi[0]", a1)
prob_left.add_boundary_condition("2", "u[2]", 0)
prob_left.add_boundary_condition("2", "psi[2]", a2)
prob_left.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob_left.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
solver_left = Solver(prob_left)
sol_left = solver_left.solve(check=False, weight=False)

corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob_top = Problem()
prob_top.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=0, deg_poly=24, spacing="linear"
)
prob_top.add_boundary_condition("1", "v[1]", 0)
prob_top.add_boundary_condition("1", "psi[1]", f"(y - 1)**2 * (y + 1)**2+{a3}")
prob_top.add_boundary_condition("3", "v[3]", 0)
prob_top.add_boundary_condition("3", "psi[3]", f"(y - 1)**2 * (y + 1)**2+{a4}")
prob_top.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob_top.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
solver_top = Solver(prob_top)
sol_top = solver_top.solve(check=False, weight=False)

# combine the solutions
a, b = 1, 1
sol_combined = a * sol_left + b * sol_top
check_flow(prob_left, sol_combined)
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
a_snap = [1.0, 1.0, 1.0]
b_snap = [0.0, 0.5, 1.0]
for a, b in zip(a_snap, b_snap):
    sol_combined = a * sol_left + b * sol_top
    an = Analysis(prob_left, sol_combined)
    fig, ax = an.plot_periodic(a, b, gapa=-1, gapb=0, n_tile=3)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"media/linear_{a:.1f}_{b:.1f}.pdf")
