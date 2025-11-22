from eom import EOM
from sympy import *
import numpy as np

class Controller:
    def __init__(
                self,
                eom: EOM,
                dt: float,
                t_max: float,
                max_motor_torque: float,
                x0: np.array = None,
                u0: np.array = None,
                ):
        """Controller for a quadrotor UAV.

        Args:
            eom (EOM): An instance of the EOM class containing the equations of motion.
            dt (float): Time step for simulation.
            t_max (float): Maximum simulation time.
            max_motor_torque (float): Maximum motor torque.
        """
        self.eom : EOM = eom # EOM object
        self.A : np.array = eom.A_num  # State matrix
        self.B : np.array = eom.B_num  # Input matrix
        self.K : np.array = None  # Gain matrix
        self.x0 : np.array = x0  # Equilibrium state
        self.u0 : np.array = u0  # Equilibrium input
        self.t_max : float = t_max  # Maximum simulation time
        self.dt : float = dt  # Time step for simulation
        self.max_motor_torque : float = max_motor_torque # Maximum motor torque

        self.states : list = []  # To store state history
        self.inputs : list = []  # To store input history
        self.ts : list = []      # To store time history


    def step(self, t: float, x: np.array, x_des: np.array) -> np.array:
        """Compute control inputs based on current and desired states.

        Args:
            x (np.array): Current state vector.
            x_des (np.array): Desired state vector.
            u_des (np.array): Desired input vector.
        Returns:
            np.array: Computed control input vector.
        """
        u = np.clip(-self.K @ (x - x_des) + self.u0, 0, self.max_motor_torque)
        xdot = self.A @ x + self.B @ u
        x = x + xdot * self.dt

        self.states.append(x)
        self.inputs.append(u)
        self.ts.append(t)
        print(f"t: {t:.3f}, x: {x}, u: {u}")
        
        
    def plot(self):
        """Plot state and input histories."""
        pass  # Implementation of plotting logic goes here