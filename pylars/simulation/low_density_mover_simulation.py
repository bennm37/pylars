"""Quasi Steady Simulation with low density Movers."""
from pylars.simulation import Simulation
from pylars import Solver
import numpy as np


class LowDensityMoverSimulation(Simulation):
    """Simulation Class for Low Density Mover Problems."""

    def __init__(self, base_problem, movers):
        super().__init__(base_problem)
        self.movers = movers
        self.name = "LowDensityMoverSimulation"
        self.solution_data = []
        self.n_movers = len(movers)
        self.mover_data = {
            "positions": [],
            "velocities": [],
            "angles": [],
            "angular_velocities": [],
        }
        self.results = {
            "solution_data": self.solution_data,
            "mover_data": self.mover_data,
        }

    def run(self, start, end, dt):
        """Run the simulation with timestep dt."""
        n_steps = np.arange(start, end, dt).shape[0]
        self.mover_array = np.zeros((4, n_steps, self.n_movers))
        super().run(start, end, dt)
        self.mover_data["positions"] = self.mover_array[0, :, :]
        self.mover_data["velocities"] = self.mover_array[1, :, :]
        self.mover_data["angles"] = self.mover_array[2, :, :]
        self.mover_data["angular_velocities"] = self.mover_array[3, :, :]
        return self.results

    def update(self, k, dt):
        """Update method for simulation."""
        for i, mover in enumerate(self.movers):
            # setup and solve the stationary, x, y and theta problems
            sol_x, force_x, torque_x = self.get_force(mover, 1, 0, 0)
            sol_y, force_y, torque_y = self.get_force(mover, 0, 1, 0)
            sol_theta, force_theta, torque_theta = self.get_force(
                mover, 0, 0, 1
            )
            sol_static, force_static, torque_static = self.get_static_force(
                mover
            )
            # solve for the velocities
            F_matrix = np.hstack([force_x, force_y, force_theta])
            torques = np.array([torque_x, torque_y, torque_theta])
            F_matrix = np.vstack([F_matrix, torques])
            F_static = np.vstack([force_static, torque_static])
            v_x, v_y, v_theta = np.linalg.solve(F_matrix, -F_static)
            v_x, v_y, v_theta = v_x[0], v_y[0], v_theta[0]
            mover.velocity = v_x + 1j * v_y
            mover.angular_velocity = v_theta
            mover.centroid += mover.velocity * self.dt
            mover.angle += mover.angular_velocity * self.dt
            self.mover_array[0, k, i] = mover.centroid
            self.mover_array[1, k, i] = mover.velocity
            self.mover_array[2, k, i] = mover.angle
            self.mover_array[3, k, i] = mover.angular_velocity
            self.solution_data.append(
                sol_static + v_x * sol_x + v_y * sol_y + v_theta * sol_theta
            )

    def get_force(self, mover, v_x, v_y, v_theta):
        """Get the force and torque on a mover with given velocity."""
        mover = mover.copy()
        mover.velocity = v_x + 1j * v_y
        mover.angular_velocity = v_theta
        prob = self.base_problem.copy()
        prob.add_mover(mover)
        sol = Solver(prob).solve()
        force = sol.force(mover.curve, mover.deriv)
        force = np.array([force.real, force.imag]).reshape(2, 1)
        torque = sol.torque(mover.curve, mover.deriv, mover.centroid)
        return sol, force, torque

    def get_static_force(self, mover):
        """Return the force on a static object.

        Overide to add alternative body forces.
        """
        return self.get_force(mover, 0, 0, 0)
