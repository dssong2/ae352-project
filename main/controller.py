from eom import EOM
from sympy import *
import numpy as np
from matplotlib import pyplot as plt

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
        
    
    def setK(self, K: np.array):
        """Set the gain matrix K.

        Args:
            K (np.array): Gain matrix.
        """
        self.K = K


    def step(self, t: float, x: np.array, x_des: np.array) -> np.array:
        # Control law in deviation coordinates
        du = -self.K @ (x - x_des)        # du can be positive or negative

        # Convert to actual motor torques (nonnegative) and saturate
        u_actual = self.u0 + du
        u_actual = np.clip(u_actual, 0.0, self.max_motor_torque)

        # After saturation, recompute the actual deviation input
        du = u_actual - self.u0

        # Linearized deviation dynamics: xdot = A x + B du
        xdot = self.A @ x + self.B @ du
        x = x + xdot * self.dt

        self.states.append(x)
        self.inputs.append(u_actual)   # log actual motor commands
        self.ts.append(t)
    
    ## Plotting functions ##
    def plot_position(self):
        """Plot position history."""
        x = np.array(self.states)[:, 0]
        y = np.array(self.states)[:, 1]
        z = np.array(self.states)[:, 2]

        plt.plot(self.ts, x, label='x')
        plt.plot(self.ts, y, label='y')
        plt.plot(self.ts, z, label='z')
        plt.title(f'Position History')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.grid()
        plt.legend()
        plt.show()
        
    
    def plot_velocity(self):
        """Plot velocity history."""
        vx = np.array(self.states)[:, 6]
        vy = np.array(self.states)[:, 7]
        vz = np.array(self.states)[:, 8]

        plt.plot(self.ts, vx, label='vx')
        plt.plot(self.ts, vy, label='vy')
        plt.plot(self.ts, vz, label='vz')
        plt.title(f'Velocity History')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid()
        plt.legend()
        plt.show()
        
        
    def plot_attitude(self):
        """Plot angular position history."""
        phi = np.array(self.states)[:, 3]
        theta = np.array(self.states)[:, 4]
        psi = np.array(self.states)[:, 5]

        plt.plot(self.ts, phi, label='phi')
        plt.plot(self.ts, theta, label='theta')
        plt.plot(self.ts, psi, label='psi')
        plt.title(f'Angular Position History')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [rad]')
        plt.grid()
        plt.legend()
        plt.show()
        
        
    def plot_angular_velocity(self):
        """Plot angular velocity history."""
        p = np.array(self.states)[:, 9]
        q = np.array(self.states)[:, 10]
        r = np.array(self.states)[:, 11]

        plt.plot(self.ts, p, label='p')
        plt.plot(self.ts, q, label='q')
        plt.plot(self.ts, r, label='r')
        plt.title(f'Angular Velocity History')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity [rad/s]')
        plt.grid()
        plt.legend()
        plt.show()
        
        
    def plot_motor_torques(self):
        """Plot motor torque history."""
        motor1 = np.array(self.inputs)[:, 0]
        motor2 = np.array(self.inputs)[:, 1]
        motor3 = np.array(self.inputs)[:, 2]
        motor4 = np.array(self.inputs)[:, 3]

        plt.plot(self.ts, motor1, label='Motor 1')
        plt.plot(self.ts, motor2, label='Motor 2')
        plt.plot(self.ts, motor3, label='Motor 3')
        plt.plot(self.ts, motor4, label='Motor 4')
        plt.title(f'Motor Torque History')
        plt.xlabel('Time [s]')
        plt.ylabel('Motor Torque [N*m]')
        plt.grid()
        plt.legend()
        plt.show()
        
    
    def plot_propeller_rpm(self):
        return # Debug this
        """Plot propeller RPM history."""
        k_f = self.eom.k_f
        motor1_rpm = np.sqrt(np.array(self.inputs)[:, 0] / k_f) * 60 / (2 * np.pi)
        motor2_rpm = np.sqrt(np.array(self.inputs)[:, 1] / k_f) * 60 / (2 * np.pi)
        motor3_rpm = np.sqrt(np.array(self.inputs)[:, 2] / k_f) * 60 / (2 * np.pi)
        motor4_rpm = np.sqrt(np.array(self.inputs)[:, 3] / k_f) * 60 / (2 * np.pi)

        plt.plot(self.ts, motor1_rpm, label='Motor 1 RPM')
        plt.plot(self.ts, motor2_rpm, label='Motor 2 RPM')
        plt.plot(self.ts, motor3_rpm, label='Motor 3 RPM')
        plt.plot(self.ts, motor4_rpm, label='Motor 4 RPM')
        plt.title(f'Propeller RPM History')
        plt.xlabel('Time [s]')
        plt.ylabel('RPM')
        plt.grid()
        plt.legend()
        plt.show()