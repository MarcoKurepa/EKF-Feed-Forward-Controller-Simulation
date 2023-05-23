import numpy as np
from scipy.integrate import solve_ivp
import controls_util as controls
from Trajectory import Trajectory
import time
from sklearn.utils import Bunch

def pad_to_shape(x, shape):
    r = np.matrix(np.zeros(shape))
    r[:x.shape[0], :x.shape[1]] = x
    return r

class TwoJointArm(object):
    def __init__(self):
        self.state = np.asmatrix(np.zeros(4)).T

        # Length of segments
        self.l1 = 46.25 * .0254
        self.l2 = 41.80 * .0254
        
        # Mass of segments
        self.m1 = 9.34 * .4536
        self.m2 = 9.77 * .4536

        # Distance from pivot to CG for each segment
        self.r1 = 21.64 * .0254
        self.r2 = 26.70 * .0254

        # Moment of inertia about CG for each segment
        self.I1 = 2957.05 * .0254*.0254 * .4536
        self.I2 = 2824.70 * .0254*.0254 * .4536

        # Gearing of each segment
        self.G1 = 140.
        self.G2 = 90.

        # Number of motors in each gearbox
        self.N1 = 1
        self.N2 = 2

        # Gravity
        self.g = 9.81

        self.stall_torque = 3.36
        self.free_speed = 5880.0 * 2.0*np.pi/60.0
        self.stall_current = 166

        self.Rm = 12.0/self.stall_current

        self.Kv = self.free_speed / 12.0
        self.Kt = self.stall_torque / self.stall_current

        # K3*Voltage - K4*velocity = motor torque
        self.K3 = np.matrix([[self.N1*self.G1, 0], [0, self.N2*self.G2]])*self.Kt/self.Rm
        self.K4 = np.matrix([[self.G1*self.G1*self.N1, 0], [0, self.G2*self.G2*self.N2]])*self.Kt/self.Kv/self.Rm

        self.voltage_log = np.matrix([0,0])
        self.current_log = np.matrix([0,0])

        self.control_law = lambda t, state: np.asmatrix(np.zeros((2, np.size(state,1))))

        pos_tol = .1
        vel_tol = .3

        self.Q = np.matrix(np.diag([1/pos_tol/pos_tol, 1/pos_tol/pos_tol, 1/vel_tol/vel_tol, 1/vel_tol/vel_tol]))
        self.R = np.matrix(np.diag([1/12.0/12.0, 1/12.0/12.0]))

        self.Q_covariance = 1e0*np.matrix(np.diag([.001**2, .001**2, .001**2, .001**2, 10.0**2, 10.0**2]))
        self.R_covariance = 1e0*np.matrix(np.diag([.01**2, .01**2]))
        self.C = np.matrix(np.block([np.identity(2), np.zeros((2,4))]))

        self.last_controller_time = -10
        self.loop_time = 1/50.0
        self.last_u = np.matrix([0,0]).T
    
    def get_ang_pos(self) -> np.matrix:
        return self.state[:2]

    def get_ang_vel(self) -> np.matrix:
        return self.state[2:4]

    def get_lin_pos(self) -> np.matrix:
        (_, end_eff) = self.fwd_kinematics(self.get_ang_pos())
        return end_eff
    
    def get_lin_joint_pos(self, ang_pos: np.matrix = None) -> np.matrix:

        if ang_pos is None:
            ang_pos = self.get_ang_pos()
        return self.fwd_kinematics(ang_pos)

    def get_lin_vel(self) -> np.matrix:
        (J, _, _) = self.jacobian(self.state)
        return J * self.get_ang_vel()

    def fwd_kinematics(self, pos: np.matrix) -> tuple[np.matrix, np.matrix]:
        [theta1, theta2] = pos.flat
        joint2 = np.matrix([self.l1*np.cos(theta1), self.l1*np.sin(theta1)]).T
        end_eff = joint2 + np.matrix([self.l2*np.cos(theta1 + theta2), self.l2*np.sin(theta1 + theta2)]).T
        return (joint2, end_eff)
    
    def inv_kinematics(self, pos: np.matrix, invert: bool = False) -> np.matrix:
        [x,y] = pos.flat
        theta2 = np.arccos((x*x + y*y - (self.l1*self.l1 + self.l2*self.l2)) / \
            (2*self.l1*self.l2))

        if invert:
            theta2 = -theta2
        
        theta1 = np.arctan2(y, x) - np.arctan2(self.l2*np.sin(theta2), self.l1 + self.l2*np.cos(theta2))
        return np.matrix([theta1, theta2]).T

    def jacobian(self, state: np.matrix) -> tuple[np.matrix, np.matrix, np.matrix]:

        pos = state[:2]
        [theta1, theta2] = pos.flat
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)

        l1 = self.l1
        l2 = self.l2
        r1 = self.r1
        r2 = self.r2

        J = np.matrix([[-l1*s1-l2*s12, -l2*s12], [l1*c1+l2*c12, l2*c12], [0,0], [0,0], [0,0], [1,1]])
        Jcm1 = np.matrix([[-r1*s1, 0], [r1*c1, 0], [0,0], [0,0], [0,0], [1,0]])
        Jcm2 = np.matrix([[-l1*s1-r2*s12, -r2*s12], [l1*c1+r2*c12, r2*c12], [0,0], [0,0], [0,0], [1,1]])

        return (J, Jcm1, Jcm2)

    def dynamics_matrices(self, state):
        [theta1, theta2, omega1, omega2] = state[:4].flat
        r1 = self.r1
        l1 = self.l1
        r2 = self.r2
        c2 = np.cos(theta2)

        hM = l1*r2*c2
        M = self.m1*np.matrix([[r1*r1, 0], [0, 0]]) + self.m2*np.matrix([[l1*l1 + r2*r2 + 2*hM, r2*r2 + hM], [r2*r2 + hM, r2*r2]]) + \
            self.I1*np.matrix([[1, 0], [0, 0]]) + self.I2*np.matrix([[1, 1], [1, 1]])

        hC = -self.m2*l1*r2*np.sin(theta2)
        C = np.matrix([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        G = self.g*np.cos(theta1) * np.matrix([self.m1*r1 + self.m2*l1, 0]).T + \
            self.g*np.cos(theta1+theta2) * np.matrix([self.m2*r2, self.m2*r2]).T

        return (M, C, G)
    
    def RK4(self, f, x, dt):
        #Perform Runge-Kutta 4 integration
        a = f(x)
        b = f(x + dt/2.0 * a)
        c = f(x + dt/2.0 * b)
        d = f(x + dt*c)
        return x + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
    
    def dynamics(self, X: np.matrix, U: np.matrix):

        omega_vec = X[2:4]
        (M, C, G) = self.dynamics_matrices(X)
        basic_torque = self.K3*U
        back_emf_loss = self.K4*omega_vec
        disturbance_torque = np.matrix([0,0]).T

        torque = basic_torque - back_emf_loss + disturbance_torque
        alpha_vec = np.linalg.inv(M)*(torque - C*omega_vec - G)
        return np.matrix(np.concatenate((omega_vec, alpha_vec)))
    
    def simulated_dynamics(self, Xhat: np.matrix, U: np.matrix):
        Xhat = Xhat.copy()
        U = U.copy()

        omega_vec = Xhat[2:4]
        (M, C, G) = self.dynamics_matrices(Xhat[:4])

        
        basic_torque = self.K3*U

        if len(Xhat) == 6:
            basic_torque += Xhat[4:]

        back_emf_loss = self.K4*omega_vec
        torque = basic_torque - back_emf_loss
        alpha_vec = M.I*(torque - C*omega_vec - G)
        dXhat = np.matrix(np.zeros_like(Xhat))
        dXhat[:4] = np.concatenate((omega_vec, alpha_vec))
        return dXhat

    def step_RK4(self, state, u, dt = .02):
        return self.RK4(lambda s: self.dynamics(s, u), state, dt)

    def step_ivp(self, state, u, dt = .02):
        step_res = solve_ivp(lambda t, s, u: self.dynamics(np.matrix(s).T, u).A1, (0, dt), state[:4], args = (u,))
        return np.matrix(step_res.y[:,-1])

    def simulated_step(self, Xhat, U, dt = .02):
        return self.RK4(lambda s: self.simulated_dynamics(s, U), Xhat, dt)
    
    def feed_forward(self, state, alpha = np.matrix([0,0]).T):
        """Determine the feed forward to satisfy given state and accelerations."""
        (D, C, G) = self.dynamics_matrices(state)
        omegas = state[2:4]
        return self.K3.I * (D*alpha + C*omegas + G + self.K4*omegas)
    
    def get_dstate(self, t, state):

        state = np.matrix(state).T
        read_state = self.sample_encoder(state)
        u = self.control_law(t, read_state)
        dstate = self.dynamics(state, u).A1
        return dstate
    
    def simulate(self, t_span, initial_state = None, t_eval = None):

        if initial_state is None:
            initial_state = self.state.A1

        sim_res = solve_ivp(self.get_dstate, t_span, initial_state, t_eval = t_eval, max_step = self.loop_time)
        return sim_res
    
    def simulate_with_ekf(self, trajectory: Trajectory, t_span, initial_state: np.matrix = None, dt = .005):

        if initial_state is None:
            initial_state = self.state
        
        (t0, tf) = t_span
        t_vec = np.arange(t0, tf + dt, dt)
        npts = len(t_vec)

        X = initial_state.copy()
        Xenc = self.sample_encoder(X)
        Xhat = initial_state.copy()

        f = self.simulated_dynamics
        ff = self.feed_forward
        ]
        KF = controls.KalmanFilter(f, ff, Xhat, self.Q_covariance[:4,:4], self.R_covariance, self.C[:,:4])

        (A, B) = self.linearize(Xhat)
        K = self.get_K(A = A, B = B)

        r = trajectory.sample(t0)[:4]
        U_ff = self.feed_forward(r)
        U = U_ff + K*(r - Xhat[:4])# - Xhat[4:]

        U = np.clip(U, -12, 12)

        X_list = np.matrix(np.zeros((4, 1)))
        Xenc_list = np.matrix(np.zeros((4, 1)))
        Xhat_list = np.matrix(np.zeros((6, 1)))
        Xerr_list = np.matrix(np.zeros((4, 1)))
        target_list = np.matrix(np.zeros((4, 1)))
        U_list = np.matrix(np.zeros((2,1)))
        U_err_list = np.matrix(np.zeros((2,1)))
        current_list = np.matrix(np.zeros((2,1)))
        Kcond_list = np.matrix(np.zeros((1,1)))
        Acond_list = np.matrix(np.zeros((1,1)))

        ignore_err = True
        downsized = True

        for t in t_vec:
            X = np.matrix(self.step_ivp(X.A1, U, dt = dt)).T
            Xenc = self.sample_encoder(X)

            KF.update(Xenc)
            
            (A, B) = self.linearize(pad_to_shape(KF.get(), (6,1)), dt = dt)
            Acond = np.linalg.cond(A)

            
            
            if Acond >= 1 and not downsized:
                print("%0.02f: downsizing" % (t))
                KF = KF.downsize(4)
                downsized = True
            elif Acond <= .001 and downsized and not ignore_err:
                print("%0.02f: upsizing" % (t))
                KF = KF.upsize(np.concatenate((KF.get(), np.matrix([0,0]).T)), self.Q_covariance)
                downsized = False
            
            if downsized:
                (A, B) = self.linearize(KF.get(), dt = dt)
                Acond = np.linalg.cond(A)
                if np.abs(Xhat[1]) <= .1:
                    print(A)
                    print(np.linalg.det(A))
            
            Acond_list = np.concatenate((Acond_list, np.matrix([Acond])), 1)
            

            K = self.get_K(KF.get())
            print(Xhat.T)
            print("A: %0.02f\tB: %0.02f\tK: %0.02f" % (np.linalg.cond(A), np.linalg.cond(B), np.linalg.cond(K)))
            r = trajectory.sample(t)[:4]
            U_ff = self.feed_forward(r)
            if downsized:
                U_err = np.matrix([0,0]).T
            else:
                U_err = KF.get()[4:]
            Xerr = r - KF.get()[:4]
            U = U_ff + K*(Xerr) - self.K3.I*U_err
            U = np.clip(U, -12, 12)

            KF.predict(U, dt)
            current = self.get_current(U, X)

            Xhat = pad_to_shape(KF.get(), (6,1))

            X_list = np.concatenate((X_list, X), 1)
            Xenc_list = np.concatenate((Xenc_list, Xenc), 1)
            Xhat_list = np.concatenate((Xhat_list, Xhat), 1)
            Xerr_list = np.concatenate((Xerr_list, Xerr), 1)
            U_list = np.concatenate((U_list, U), 1)
            U_err_list = np.concatenate((U_err_list, U_err), 1)
            current_list = np.concatenate((current_list, current), 1)
            Kcond_list = np.concatenate((Kcond_list, np.matrix([np.linalg.cond(K)])), 1)
            target_list = np.concatenate((target_list, r), 1)

        return Bunch(t = t_vec, X = X_list, Xenc = Xenc_list, \
            Xhat = Xhat_list, U = U_list, U_err = U_err_list, \
            current = current_list, Xerr = Xerr_list, \
            Kcond = Kcond_list, Acond = Acond_list, target = target_list)
    
    def get_voltage_log(self, sim_solution):
        return self.control_law(sim_solution.t, sim_solution.y)

    def get_current(self, voltage, state):
        omegas = state[2:4].copy()
        omegas[0,:] = omegas[0,:] * self.G1
        omegas[1,:] = omegas[1,:] * self.G2
        stall_voltage = voltage - omegas/self.Kv
        current = stall_voltage/self.Rm
        return current

    def get_current_log(self, sim_solution, voltage_log = None):
        if voltage_log is None:
            voltage_log = self.get_voltage_log(sim_solution)
        return self.get_current(voltage_log, sim_solution.y)
    
    
    def linearize(self, state = None, eps = 1e-4, dt = .005):
        if state is None:
            state = self.state

        state = state.copy()
        
        f = lambda x, u: self.simulated_dynamics(x, u)

        (Ac,Bc) = controls.linearize(f, x = state, u = self.feed_forward(state), eps = eps)
        (A, B) = controls.discretize_ab(Ac, Bc, dt)

   

        return (A,B)

    
    def get_K(self, state = None, A = None, B = None):
        if A is None or B is None:
            if state is None:
                state = self.state
            (A,B) = self.linearize(state)

        K = controls.lqr(A[:4,:4], B[:4,:4], self.Q, self.R)
        return K

    def sample_encoder(self, X):
        return X.copy() + pad_to_shape(np.matrix(np.random.normal(0,.005,4)).T, X.shape)
   
