"""Flow a domain with a cardiod interior curve.""" ""
from pylars import Problem, Solver, Analysis
from pylars.colormaps import parula
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import numpy as np
from scipy.io import loadmat


def solve_cardioid():
    """Solve the Cardioid flow problem."""
    prob = Problem()
    corners = [-3 - 1j, 3 - 1j, 3 + 1j, -3 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=1000,
        num_poles=1,
        deg_poly=120,
        spacing="linear",
    )

    cardioid = lambda t: 0.2 * ((np.exp(2j * np.pi * t) - 1) ** 2 - 1)

    prob.add_interior_curve(
        cardioid,
        num_points=250,
        deg_laurent=80,
        centroid=0.0 + 0.0j,
        aaa=False,
        aaa_mmax=150,
    )
    prob.domain.plot()
    plt.tight_layout()
    plt.show()
    p_drop = 37
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "psi[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "p[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", p_drop)
    prob.add_boundary_condition("3", "v[3]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)

    solver = Solver(prob, verbose=True)
    sol = solver.solve(check=False, normalize=False, weight=False)
print(f"Error: {solver.max_error}")
    return prob, sol, cardioid


def plot_moffat_eddies(prob, sol, cardioid):
    """Plot Moffat eddies from Xue et al. 2023."""
    # recreate Moffat eddies from Xue et al. 2023
    # nc -> near_cusp
    delta = 0.1
    y_sf = 0.5
    resolution = 201
    x_nc = np.linspace(-0.20 - delta, -0.2, resolution)
    y_nc = np.linspace(-y_sf * delta, y_sf * delta, resolution)
    X_nc, Y_nc = np.meshgrid(x_nc, y_nc)
    Z_nc = X_nc + 1j * Y_nc
    inside = prob.domain.mask_contains(Z_nc)
    Z_nc[~inside] = np.nan
    psi_nc = sol.psi(Z_nc).reshape(X_nc.shape)
    fig, ax = plt.subplots()
    psi_c = sol.psi(-0.2)
    levels = np.logspace(-11.9, -5, 30)
    ax.contour(
        X_nc,
        Y_nc,
        np.abs(psi_nc - psi_c),
        locator=LogLocator(),
        levels=levels,
        cmap=parula,
    )
    norm = LogNorm(vmin=levels[0], vmax=levels[-1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=parula)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="$|\psi_c-\psi|$",
    )
    plt.plot(
        cardioid(np.linspace(0, 1, 1000)).real,
        cardioid(np.linspace(0, 1, 1000)).imag,
        c="k",
    )
    ax.axis("equal")
    ax.axis("off")
    ax.set(xlim=(-0.2 - delta, -0.2), ylim=(-y_sf * delta, y_sf * delta))
    return fig, ax


def plot_matlab_error(sol):
    """Plot error against MATLAB for a cardioid domain."""
    # plot error against the MATLAB solution
    data = loadmat("data/cardioid/data.mat")
    zz = data["zz"]
    inside = ~sol.problem.domain.mask_contains(zz)
    matlab_psi_data = data["psi_data"]
    matlab_psi_data[inside] = np.nan
    matlab_omega_data = data["omega_data"]
    matlab_omega_data[inside] = np.nan
    matlab_p_data = data["p_data"]
    matlab_p_data[inside] = np.nan
    matlab_u_data = data["u_data"]
    matlab_u_data[inside] = np.nan

    pylars_psi_data = sol.psi(zz).reshape(zz.shape)
    pylars_psi_data[inside] = np.nan
    pylars_omega_data = sol.omega(zz).reshape(zz.shape)
    pylars_omega_data[inside] = np.nan
    pylars_p_data = sol.p(zz).reshape(zz.shape)
    pylars_p_data[inside] = np.nan
    pylars_u_data = sol.uv(zz).reshape(zz.shape)
    pylars_u_data[inside] = np.nan

    # plot error
    fig, ax = plt.subplots(2, 2, figsize=(6, 4))
    cmap = plt.get_cmap("RdBu_r")
    cmap.set_bad("white")
    psi_diff = np.abs(matlab_psi_data - pylars_psi_data)
    psi_diff_max = psi_diff[~inside].max()
    print(f"Max error in psi: {psi_diff_max}")
    im00 = ax[0, 0].imshow(
        psi_diff,
        cmap=cmap,
        norm=LogNorm(vmin=psi_diff[~inside].min() + 1e-17, vmax=psi_diff_max),
    )
    ax[0, 0].axis("off")
    plt.text(
        0.5,
        -0.1,
        "(a) Difference in $\psi$",
        ha="center",
        va="center",
        transform=ax[0, 0].transAxes,
    )
    plt.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)
    p_diff = np.abs(matlab_p_data - pylars_p_data)
    p_diff_max = p_diff[~inside].max()
    print(f"Max error in p: {p_diff_max}")
    im10 = ax[1, 0].imshow(
        p_diff,
        cmap=cmap,
        norm=LogNorm(vmin=p_diff[~inside].min() + 1e-17, vmax=p_diff_max),
    )
    ax[1, 0].axis("off")
    plt.text(
        0.5,
        -0.1,
        "(c) Difference in $p$",
        ha="center",
        va="center",
        transform=ax[1, 0].transAxes,
    )
    plt.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)
    omega_diff = np.abs(matlab_omega_data - pylars_omega_data)
    omega_diff_max = omega_diff[~inside].max()
    print(f"Max error in omega: {omega_diff_max}")
    im01 = ax[0, 1].imshow(
        omega_diff,
        cmap=cmap,
        norm=LogNorm(
            vmin=omega_diff[~inside].min() + 1e-17,
            vmax=omega_diff_max,
        ),
    )
    ax[0, 1].axis("off")
    plt.text(
        0.5,
        -0.1,
        "(b) Difference in $\omega$",
        ha="center",
        va="center",
        transform=ax[0, 1].transAxes,
    )
    plt.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)
    u_diff = np.abs(matlab_u_data - pylars_u_data)
    u_diff_max = u_diff[~inside].max()
    print(f"Max error in u: {u_diff_max}")
    im11 = ax[1, 1].imshow(
        u_diff,
        cmap=cmap,
        norm=LogNorm(vmin=u_diff[~inside].min() + 1e-17, vmax=u_diff_max),
    )
    ax[1, 1].axis("off")
    plt.text(
        0.5,
        -0.1,
        "(d) Difference in $|\mathbf{u}|$",
        ha="center",
        va="center",
        transform=ax[1, 1].transAxes,
    )
    plt.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.95)
    return fig, ax


if __name__ == "__main__":
    prob, sol, cardioid = solve_cardioid()
    # an = Analysis(sol)
    # fig, ax = an.plot(resolution=301, interior_patch=True, enlarge_patch=1.0)
    # plt.show()
    # plt.savefig("media/cardioid.pdf", bbox_inches="tight")
    # plt.show()
    fig, ax = plot_moffat_eddies(prob, sol, cardioid)
    plt.savefig("media/moffat_eddies.pdf", bbox_inches="tight")
    # fig, ax = plot_matlab_error(sol)
    # plt.savefig("media/matlab_error.pdf", bbox_inches="tight")
