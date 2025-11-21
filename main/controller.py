from eom import EOM
from sympy import *
import numpy as np

class Controller:
    def __init__(self, eom: EOM):
        """Controller for a quadrotor UAV.

        Args:
            eom (EOM): An instance of the EOM class containing the equations of motion.
        """
        self.eom = eom
        self.Kp_pos = np.diag([1.0, 1.0, 1.0])  # Proportional gain for position
        self.Kd_pos = np.diag([0.5, 0.5, 0.5])  # Derivative gain for position
        self.Kp_att = np.diag([1.0, 1.0, 1.0])  # Proportional gain for attitude
        self.Kd_att = np.diag([0.5, 0.5, 0.5])  # Derivative gain for attitude
        
    def run(self, x: np.array, x_des: np.array, u_des: np.array) -> np.array:
        """Compute control inputs based on current and desired states.

        Args:
            x (np.array): Current state vector.
            x_des (np.array): Desired state vector.
            u_des (np.array): Desired input vector.
        Returns:
            np.array: Computed control input vector.
        """

        while t < self.t_estimated_apogee:
            print(f"t: {t:.3f}, xhat: {xhat}, u: {np.rad2deg(u)}")
            A = np.array(self.A.n()).astype(np.float64)
            B = np.array(self.B.n()).astype(np.float64)

            # Gain scheduling based on vertical velocity
            K = self.control_law(xhat, t)
            u = np.clip(-K @ (xhat - self.x0) + self.u0, np.deg2rad(-self.max_delta), np.deg2rad(self.max_delta))
            # u = np.array([0.0])  # For testing, set aileron to 0
            
            ## Control Law ##
            theta, phi, psi = self.quat_to_euler_xyz(xhat[6:10])  # Convert quaternion to Euler angles
            y = self.deriveSensorModels(t, xhat[0], xhat[1], xhat[2],
                                    theta, phi, psi)  # Simulated sensor measurements
            
            ## Add back thrust and gravity terms (differentiated to 0 in computing A) ##
            xdot = A @ xhat + B @ u + self.get_thrust_accel(t) + self.get_gravity_accel(xhat) \
                    # - self.L @ (C @ xhat - y)
            xhat = xhat + xdot * self.dt
            xhat[6:10] /= np.linalg.norm(xhat[6:10])
            
            self.states.append(xhat)
            self.inputs.append(u)
            t = t + self.dt