from eom import EOM
from sympy import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
        self.des_states = []
         
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
        self.des_states.append(x_des.copy())
    
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
    
    def plot_path(self, elev=90, azim=-90, max_points=1000):
        """Plot the 3D path with a time-based color gradient (downsampled)."""
        if len(self.states) == 0:
            print("No states logged; nothing to plot.")
            return

        states = np.array(self.states)
        t = np.array(self.ts)

        # -----------------------------
        # Downsample if too many points
        # -----------------------------
        N = len(states)
        if N > max_points:
            idx = np.linspace(0, N - 1, max_points).astype(int)
            states = states[idx]
            t = t[idx]

        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]

        # Normalize time for color mapping
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)

        # Make colored segments
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cmap = plt.get_cmap("turbo")  # or "nipy_spectral", etc.
        lc = Line3DCollection(segments, cmap=cmap)
        lc.set_array(t_norm)
        lc.set_linewidth(2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(lc)

        # Markers
        ax.scatter(x[0], y[0], z[0], c="green", s=40, label="start")
        ax.scatter(x[-1], y[-1], z[-1], c="red", s=40, label="end")

        # Labels & style
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("3D Trajectory Path (Colored by Time)")
        ax.legend()
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True)

        # Equal scaling
        max_range = np.array([x.max() - x.min(),
                            y.max() - y.min(),
                            z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Colorbar with REAL time labels
        cbar = fig.colorbar(lc, ax=ax, pad=0.1)
        ticks = np.linspace(0, 1, 5)
        tick_labels = np.linspace(t.min(), t.max(), 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v:.1f}s" for v in tick_labels])
        cbar.set_label("Time [s]")

        plt.tight_layout()
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
    
    def plot_track(self, flight_id=1, dt_eval=0.5):
        """
        Generic trajectory tracking plot and metrics.
        
        For all flights:
        - position error vs time
        - yaw error vs time
        - global RMS error

        For flight_id == 3:
        - checks 5m leg timing (should be ~5s)
        - checks 5m leg distances (should be ~5m)
        - checks yaw = +90° requirement (final and segment RMS)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if len(self.states) == 0 or not hasattr(self, "des_states") or len(self.des_states) == 0:
            print("No logged desired states; run a flight first.")
            return

        states = np.array(self.states)
        des    = np.array(self.des_states)
        ts     = np.array(self.ts)

        # --- basic errors ---
        pos     = states[:, 0:3]
        pos_des = des[:,    0:3]
        err_pos = pos - pos_des
        err_norm = np.linalg.norm(err_pos, axis=1)

        yaw     = states[:, 3]
        yaw_des = des[:,    3]
        yaw_err = (yaw - yaw_des + np.pi) % (2*np.pi) - np.pi

        # --- downsample for plotting ---
        dt_sim = ts[1] - ts[0]
        stride = max(1, int(dt_eval / dt_sim))

        ts_s   = ts[::stride]
        err_s  = err_norm[::stride]
        yaw_s  = yaw_err[::stride]

        # --- plots ---
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,6))

        ax[0].plot(ts_s, err_s)
        ax[0].set_ylabel("‖pos error‖ [m]")
        ax[0].set_title(f"Flight {flight_id} Tracking Error")
        ax[0].grid(True)

        ax[1].plot(ts_s, yaw_s)
        ax[1].set_ylabel("yaw error [rad]")
        ax[1].set_xlabel("time [s]")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

        # --- global RMS ---
        rms_pos = np.sqrt(np.mean(err_norm**2))
        rms_yaw = np.sqrt(np.mean(yaw_err**2))

        print(f"Global RMS pos error:  {rms_pos:.3f} m")
        print(f"Global RMS yaw error:  {rms_yaw:.3f} rad")

        # --- Flight 3 requirement checks ---
        if flight_id == 3:
            self._flight3_checks(ts, states, des)

    def _flight3_checks(self, ts, states, des):
        """
        Internal helper: Flight 3 mission requirement checks.
        Uses known waypoints:
        Leg 1: (0,0,1) -> (5,0,1)
        Leg 2: (5,0,1) -> (5,5,1)
        Desired yaw = +90° after yaw segment.
        """
        import numpy as np

        states = np.array(states)
        des    = np.array(des)
        ts     = np.array(ts)

        pos     = states[:, 0:3]
        pos_des = des[:,    0:3]
        yaw     = states[:, 3]
        yaw_des = des[:,    3]

        def find_idx_first(point, tol=1e-2):
            d = np.linalg.norm(pos_des - np.array(point), axis=1)
            idx = np.where(d < tol)[0]
            if len(idx) == 0:
                return int(np.argmin(d))
            return int(idx[0])

        def find_idx_last(point, tol=1e-2):
            d = np.linalg.norm(pos_des - np.array(point), axis=1)
            idx = np.where(d < tol)[0]
            if len(idx) == 0:
                return int(np.argmin(d))
            return int(idx[-1])

        print("\n--- Flight 3 Mission Checks ---")

        # ---- Leg 1: (0,0,1) -> (5,0,1) ----
        # start AFTER ascent+hover: last time at (0,0,1)
        start1 = find_idx_last([0.0, 0.0, 1.0])
        # end at first arrival at (5,0,1)
        end1   = find_idx_first([5.0, 0.0, 1.0])

        if end1 > start1:
            t_leg1 = ts[end1] - ts[start1]
            d_leg1 = np.linalg.norm(pos[end1] - pos[start1])
            print(f"Leg 1 (5m) duration: {t_leg1:.3f} s (target ~5 s)")
            print(f"Leg 1 distance:      {d_leg1:.3f} m (target = 5 m)")
        else:
            print("Warning: Could not identify 1st leg indices")

        # ---- Leg 2: (5,0,1) -> (5,5,1) ----
        # start AFTER hover+yaw: last time at (5,0,1)
        start2 = find_idx_last([5.0, 0.0, 1.0])
        # end at first arrival at (5,5,1)
        end2   = find_idx_first([5.0, 5.0, 1.0])

        if end2 > start2:
            t_leg2 = ts[end2] - ts[start2]
            d_leg2 = np.linalg.norm(pos[end2] - pos[start2])
            print(f"Leg 2 (5m) duration: {t_leg2:.3f} s (target ~5 s)")
            print(f"Leg 2 distance:      {d_leg2:.3f} m (target = 5 m)")

            # yaw tracking on leg 2
            yaw_err_seg2 = (yaw[start2:end2] - yaw_des[start2:end2] + np.pi) % (2*np.pi) - np.pi
            rms_yaw_leg2 = np.sqrt(np.mean(yaw_err_seg2**2))
            print(f"RMS yaw error on leg 2: {rms_yaw_leg2:.3f} rad")
        else:
            print("Warning: Could not identify 2nd leg indices")

        # ---- Final yaw check ----
        yaw_final_err = (yaw[-1] - yaw_des[-1] + np.pi) % (2*np.pi) - np.pi
        print(f"Final yaw target: {yaw_des[-1]:.3f} rad (≈ +90°)")
        print(f"Final yaw actual: {yaw[-1]:.3f} rad")
        print(f"Final yaw error:  {yaw_final_err:.3f} rad")

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
        
        
    def plot_all(self):
        self.plot_position()
        self.plot_path()
        self.plot_velocity()
        self.plot_attitude()
        self.plot_angular_velocity()
        self.plot_motor_torques()
        self.plot_propeller_rpm()