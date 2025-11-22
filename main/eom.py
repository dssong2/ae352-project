from sympy import *
import numpy as np

class EOM:
    def __init__(self):
        """Equations of Motion for a quadrotor UAV."""
        self.mass : float = None         # mass (scalar)
        self.inertia : np.array = None   # inertia (3x3 matrix or 3-vector)
        self.leg_length : float = None   # leg length (scalar)
        self.g : float = 9.81            # gravitational acceleration (scalar)
        self.k_f : float = None          # thrust constant (scalar)
        self.k_yaw : float = None        # yaw torque constant (scalar)
        self.params : dict = {}          # dictionary mapping symbolic params to numeric values
        
        # placeholders for symbolic stuff
        self.x_sym : Matrix = None      # state vector symbols
        self.u_sym : Matrix = None      # input vector symbols
        self.p_sym : Matrix = None      # parameter symbols
        self.f_sym : Matrix = None      # f(x,u,p) symbolic EOM
        self.f_num : Matrix = None      # f(x,u,p) numeric EOM
        
        self.m_e : dict = None  # equilibrium mass
        self.n_e : dict = None  # equilibrium input
        
        self.A_sym : Matrix = None  # state matrix
        self.B_sym : Matrix = None  # input matrix
        self.A_subs : Matrix = None  # state matrix at equilibrium
        self.B_subs : Matrix = None  # input matrix at equilibrium
        self.A_num : np.array = None  # numeric state matrix
        self.B_num : np.array = None  # numeric input matrix
        
    
    def define_symbols(self):
        """Define symbolic variables for states, inputs, and parameters.
        """

        m, Jx, Jy, Jz, l, k_f, g, k_yaw = symbols('m Jx Jy Jz l k_f g k_yaw', real=True, positive=True)
        self.p_sym = Matrix([m, Jx, Jy, Jz, l, k_f, g, k_yaw])
        
        # -----------------------------
        # State symbols
        # -----------------------------
        # position in WORLD frame
        p_x, p_y, p_z = symbols('p_x p_y p_z', real=True)

        # yaw, pitch, roll (psi, theta, phi)
        psi, theta, phi = symbols('psi theta phi', real=True)

        # linear velocity in BODY frame
        v_x, v_y, v_z = symbols('v_x v_y v_z', real=True)

        # angular velocity in BODY frame
        w_x, w_y, w_z = symbols('w_x w_y w_z', real=True)

        # -----------------------------
        # Inputs (motor torques)
        # -----------------------------
        tau1, tau2, tau3, tau4 = symbols('tau1 tau2 tau3 tau4', real=True)
        
        self.x_sym = Matrix([
            p_x, p_y, p_z,
            psi, theta, phi,
            v_x, v_y, v_z,
            w_x, w_y, w_z,
        ])

        self.u_sym = Matrix([tau1, tau2, tau3, tau4])
    
    def set_parameters(self,
                 mass: float,
                 inertia: np.array,
                 leg_length: float,
                 k_f_val: float,
                 k_yaw_val: float):
        """Set numeric parameters for the EOMs.

        Args:
            mass (float): mass of the quadrotor
            inertia (np.array): inertia matrix or vector of the quadrotor
            leg_length (float): length of the quadrotor legs
            k_f_val (float): thrust constant
            k_yaw_val (float): yaw torque constant
        """
        if self.p_sym is None:
            self.define_symbols()
        
        m, Jx, Jy, Jz, l, k_f, g, k_yaw = self.p_sym
        self.mass = mass
        self.inertia = inertia
        self.leg_length = leg_length
        self.k_f = k_f_val
        self.k_yaw = k_yaw_val
        
        self.params = {
            m: mass,
            Jx: inertia[0, 0] if inertia.shape == (3, 3) else inertia[0],
            Jy: inertia[1, 1] if inertia.shape == (3, 3) else inertia[1],
            Jz: inertia[2, 2] if inertia.shape == (3, 3) else inertia[2],
            l: leg_length,
            k_f: k_f_val,
            k_yaw: k_yaw_val,
            g: Float(9.81),
        }


    def derive_eoms_symbolic(self):
        """Derive the symbolic equations of motion for the quadrotor UAV.
        """
        if self.x_sym is None or self.u_sym is None:
            self.define_symbols()

        # -----------------------------
        # Unpack state and input symbols
        # -----------------------------
        p_x, p_y, p_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z = self.x_sym
        v_in_body = Matrix([v_x, v_y, v_z])
        w_in_body = Matrix([w_x, w_y, w_z])

        # -----------------------------
        # Unpack input symbols
        # -----------------------------
        tau1, tau2, tau3, tau4 = self.u_sym

        # -----------------------------
        # Unpack parameter symbols
        # -----------------------------
        m, Jx, Jy, Jz, l, k_f, g, k_yaw = self.p_sym

        J : Matrix = diag(Jx, Jy, Jz)

        # -----------------------------
        # Rotation matrices (body -> world)
        # -----------------------------
        Rz = Matrix([
            [cos(psi), -sin(psi), 0],
            [sin(psi),  cos(psi), 0],
            [0, 0, 1],
        ])
        Ry = Matrix([
            [cos(theta), 0, sin(theta)],
            [ 0, 1, 0],
            [-sin(theta), 0, cos(theta)],
        ])
        Rx = Matrix([
            [1, 0, 0],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi), cos(phi)],
        ])

        R_body_in_world : Matrix = Rz @ Ry @ Rx

        # -----------------------------
        # Angular velocity -> Euler angle rates
        # (using your original mapping construction)
        # -----------------------------
        ex = Matrix([[1], [0], [0]])
        ey = Matrix([[0], [1], [0]])
        ez = Matrix([[0], [0], [1]])

        M = simplify(
            Matrix.hstack((Ry @ Rx).T @ ez, Rx.T @ ey, ex).inv(),
            full=True
        )

        # -----------------------------
        # Individual rotor thrusts from motor torques
        # fz_n = tau_n * k_f   (k_f = b/a)
        # -----------------------------
        F1 = k_f * tau1
        F2 = k_f * tau2
        F3 = k_f * tau3
        F4 = k_f * tau4

        # Net thrust along body z
        Fz_total = F1 + F2 + F3 + F4

        # -----------------------------
        # Forces in body frame
        # -----------------------------
        # gravity in world, then in body
        F_grav_world = Matrix([[0], [0], [-m * g]])
        F_grav_body  = R_body_in_world.T @ F_grav_world

        # thrust in body frame (along +z_body)
        F_thrust_body = Matrix([[0], [0], [Fz_total]])

        # total applied force in body
        f_in_body = F_grav_body + F_thrust_body

        # -----------------------------
        # Body torques from arm length and rotor forces (plus config)
        # -----------------------------
        # roll (x):  tau_x = l (F2 - F4)
        # pitch (y): tau_y = l (F3 - F1)
        # yaw (z):   tau_z from differential motor torque
        #
        # Simple yaw model: reaction yaw torque proportional to motor torque
        #   tau_z ∝ (-tau1 + tau2 - tau3 + tau4)

        tau_x_expr = l * (F2 - F4)
        tau_y_expr = l * (F3 - F1)
        tau_z_expr = k_yaw * (-tau1 + tau2 - tau3 + tau4)

        tau_in_body = Matrix([[tau_x_expr],
                              [tau_y_expr],
                              [tau_z_expr]])

        # -----------------------------
        # Equations of motion
        # -----------------------------
        # position dynamics in WORLD frame
        p_dot = R_body_in_world @ v_in_body  # [p_x_dot, p_y_dot, p_z_dot]^T

        # attitude kinematics (psi, theta, phi)
        angles_dot = M @ w_in_body           # [psi_dot, theta_dot, phi_dot]^T

        # translational dynamics in BODY frame
        ## Check this equation later, coriolis force term included
        v_dot = (1 / m) * (f_in_body - w_in_body.cross(m * v_in_body))

        # rotational dynamics in BODY frame
        w_dot = J.inv() @ (tau_in_body - w_in_body.cross(J @ w_in_body))

        # -----------------------------
        # Pack state and dynamics
        # -----------------------------
        x = Matrix([
            p_x, p_y, p_z,
            psi, theta, phi,
            v_x, v_y, v_z,
            w_x, w_y, w_z,
        ])

        u = Matrix([tau1, tau2, tau3, tau4])

        f = Matrix.vstack(
            p_dot,        # 3x1
            angles_dot,   # 3x1
            v_dot,        # 3x1
            w_dot,        # 3x1
        )

        f = simplify(f, full=True)

        # store symbolic structures on the object
        self.x_sym = x
        self.u_sym = u
        self.f_sym = f


    def derive_eoms_numeric(self):
        """Evaluate the symbolic equations of motion at the given numeric parameters.

        Raises:
            ValueError: if symbolic EOMs have not been derived yet.
        """
        if self.f_sym is None:
            raise ValueError("Symbolic EOMs must be derived first.")

        self.f_num = self.f_sym.subs(self.params).evalf()
        
    
    def get_equilibrium(self):
        """Compute the equilibrium state and input for hovering. Sets self.m_e and self.n_e.
        """

        if self.f_num is None:
            raise ValueError("Numeric EOMs must be derived first.")

        p_x, p_y, p_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z = self.x_sym
        tau1, tau2, tau3, tau4 = self.u_sym
        
        m_e = {
            p_x: 0.,
            p_y: 0.,
            p_z: 0.,
            psi: 0.,
            theta: 0.,
            phi: 0.,
            v_x: 0.,
            v_y: 0.,
            v_z: 0.,
            w_x: 0.,
            w_y: 0.,
            w_z: 0.,
        }

        tau_e = self.mass * self.g / (4 * self.k_f)
        n_e = {
            tau1: tau_e,
            tau2: tau_e,
            tau3: tau_e,
            tau4: tau_e,
        }

        self.m_e = m_e
        self.n_e = n_e
        
    
    def get_AB(self):
        """Compute the linearized state-space matrices A and B around the equilibrium point.
        Sets purely symbolic and numeric A and B matrices, symbolic A and B matrices substituted at equilibrium ,
        and numeric A and B matrices substituted at equilibrium.
        """
        if self.f_num is None:
            raise ValueError("Numeric EOMs must be derived first.")
        if self.m_e is None or self.n_e is None:
            raise ValueError("Equilibrium point must be computed first.")

        x = self.x_sym
        u = self.u_sym

        self.A_sym = self.f_num.jacobian(x)
        self.B_sym = self.f_num.jacobian(u)

        self.A_subs = simplify(self.A_sym.subs({**self.m_e, **self.n_e}), full=True)
        self.B_subs = simplify(self.B_sym.subs({**self.m_e, **self.n_e}), full=True)
        
        self.A_num = np.array(self.A_subs).astype(np.float64)
        self.B_num = np.array(self.B_subs).astype(np.float64)
        
    def setup(self):
        """Run the full setup: derive symbolic EOMs, derive numeric EOMs, compute equilibrium, and get A and B matrices.
        """
        if self.params == {}:
            raise ValueError("Numeric parameters must be set before full setup.")
        self.define_symbols()
        self.derive_eoms_symbolic()
        self.derive_eoms_numeric()
        self.get_equilibrium()
        self.get_AB()