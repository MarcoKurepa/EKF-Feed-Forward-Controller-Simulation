import numpy as np
from scipy.linalg import solve_discrete_are, expm
import time

def RK4(f, x, dt): #Runge-Kutta 4 Integration
    a = f(x)
    b = f(x + dt/2.0 * a)
    c = f(x + dt/2.0 * b)
    d = f(x + dt*c)
    return x + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0

def linearize(f, x: np.matrix, u: np.matrix, eps = 1e-4) -> tuple[np.matrix, np.matrix]:

    jac_x = jacobian_with_x(f, x, u, eps)
    jac_u = jacobian_with_u(f, x, u, eps)

    return (jac_x, jac_u)

def discretize_ab(A, B, dt):
    states = A.shape[0]
    inputs = B.shape[1]

    M = np.matrix(expm(np.block([[A, B], [np.zeros((inputs, states)), np.zeros((inputs, inputs))]]) * dt))
    return M[:states, :states], M[:states, states:]

def jacobian_with_x(f, x: np.matrix, u: np.matrix, eps = 1e-4) -> np.matrix:
    """Get the jacobian of f with respect to x evaluated at x, u"""
    Jx = np.matrix(np.zeros((x.shape[0], x.shape[0])))

    for i in range(x.shape[0]):
        ux = x.copy()
        ux[i] += eps        
        lx = x.copy()
        lx[i] -= eps    
        col = (f(ux, u) - f(lx, u)) / (2*eps)
        Jx[:,i] = col

    return Jx

def jacobian_with_u(f, x: np.matrix, u:np.matrix, eps = 1e-4) -> np.matrix:
    """Get the jacobian of f with respect to u evaluated at x, u"""
    Ju = np.matrix(np.zeros((x.shape[0], u.shape[0])))

    for i in range(u.shape[0]): 
        uu = u.copy()
        uu[i] += eps        
        lu = u.copy()
        lu[i] -= eps          
        Ju[:,i] = (f(x, uu) - f(x, lu)) / (2*eps)

    return Ju

def lqr(A: np.matrix, B: np.matrix, Q: np.matrix, R: np.matrix) -> np.matrix:
    S = solve_discrete_are(A, B, Q, R)
    return (R + B.T * S * B).I * B.T * S * A

class KalmanFilter(object):
    def __init__(self, f, ff, Xhat, Q, R, C):
        self.f = f
        self.ff = ff
        self.Xhat = Xhat.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.C = C.copy()
        
        self.P = Q.copy()
        self.nstates = Xhat.shape[0]
        self.noutputs = R.shape[0]

        self.A = None
        self.B = None
    
    def get(self):
        return self.Xhat.copy()

    def update(self, X_measured):
        Kal = self.P*self.C.T*np.linalg.pinv(self.C*self.P*self.C.T + self.R)
        self.Xhat = self.Xhat + Kal*(X_measured[:self.noutputs] - self.C*self.Xhat)
        self.P = (np.identity(self.nstates) - Kal*self.C)*self.P

    def predict(self, U, dt):
        self.Xhat = RK4(lambda x: self.f(x,U), self.Xhat, dt)
        self.relinearize(dt)
        self.P = self.A*self.P*self.A.T + self.Q

    def downsize(self, nstates):
        newKF = KalmanFilter(self.f, self.ff, self.Xhat[:nstates], self.Q[:nstates,:nstates], self.R, self.C[:,:nstates])
        newKF.P = self.P[:nstates, :nstates]
        return newKF

    def upsize(self, Xhat, Q):
        new_nstates = Xhat.shape[0]
        new_Xhat = np.concatenate((self.Xhat, Xhat[self.nstates:]))
        new_Q = Q.copy()
        new_Q[:self.nstates, :self.nstates] = self.Q
        new_C = np.concatenate((self.C, np.matrix(np.zeros((self.noutputs, new_nstates - self.nstates)))), 1)
        newKF = KalmanFilter(self.f, self.ff, new_Xhat, new_Q, self.R, new_C)
        newKF.P[:self.nstates, :self.nstates] = self.P
        return newKF

    def relinearize(self, dt):
        (Ac, Bc) = linearize(self.f, self.Xhat, self.ff(self.Xhat))
        (self.A, self.B) = discretize_ab(Ac, Bc, dt)
