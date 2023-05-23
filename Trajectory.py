from __future__ import annotations
import numpy as np

class Trajectory(object):

    def __init__(self, times: np.matrix, states: np.matrix) -> Trajectory:
        #Initialize a Trajectory
       
        self.times: np.matrix = times
        self.states: np.matrix = states
        self.start_time = times[0]
        self.end_time = times[-1]
    
    def clip_time(self, time: float) -> np.matrix:
        return np.matrix(np.clip(time, self.start_time, self.end_time))

    def sample(self, time: float) -> np.matrix:
time: time to sample
        time = self.clip_time(time)
        prev_idx = np.where(self.times <= time)[0][-1]
        next_idx = np.where(self.times >= time)[0][0]

        if prev_idx == next_idx:
            return self.states[:, prev_idx]
        
        prev_val = self.states[:, prev_idx]
        next_val = self.states[:, next_idx]
        prev_time = self.times[prev_idx]
        next_time = self.times[next_idx]

        return (next_val - prev_val)/(next_time - prev_time)*(time-prev_time) + prev_val
    
    def append(self, other: Trajectory) -> Trajectory:

        # Create new trajectory based off of this one
        combined = Trajectory(self.times, self.states)
        # Adjust timestamps on other trajectory
        other.times = other.times + combined.end_time - other.start_time
        # Combine the time and states
        combined.times = np.concatenate((combined.times, other.times[1:]))
        combined.states = np.concatenate((combined.states, other.states[:,1:]), 1)
        # Update the end time
        combined.end_time = max(combined.times)
        return combined
    
    
    def to_table(self) -> np.ndarray:
        return np.concatenate((self.times, self.states.T), 1)

def from_coeffs(coeffs: np.matrix, t0, tf, n = 100) -> Trajectory:

        order = np.size(coeffs, 0) - 1
        t = np.matrix(np.linspace(t0, tf, n)).T
        pos_t_vec = np.power(t, np.arange(order + 1))
        pos_vec = pos_t_vec * coeffs
        vel_t_vec = np.concatenate((np.zeros((n,1)), np.multiply(pos_t_vec[:, 0:-1], np.repeat(np.array([np.arange(order) + 1]), n, 0))), 1)
        vel_vec = vel_t_vec * coeffs
        acc_t_vec = np.concatenate((np.zeros((n,2)), np.multiply(vel_t_vec[:, 1:-1], np.repeat(np.array([np.arange(order - 1) + 2]), n, 0))), 1)
        acc_vec = acc_t_vec * coeffs

        states = np.asmatrix(np.concatenate((pos_vec, vel_vec, acc_vec), 1).T)
        return Trajectory(t, states)

def interpolate_states(t0, tf, state0, statef):
    coeffs = __cubic_interpolation(t0, tf, state0, statef)
    return from_coeffs(coeffs, t0, tf)


def __cubic_interpolation(t0, tf, state0: np.matrix, statef: np.matrix) -> np.matrix:

    pos_row = lambda t: np.matrix([1, t, t*t, t*t*t])
    vel_row = lambda t: np.matrix([0, 1, 2*t, 3*t*t])

    rhs = np.concatenate((state0.reshape((2,2)), statef.reshape(2,2)))
    lhs = np.concatenate((pos_row(t0), vel_row(t0), pos_row(tf), vel_row(tf)))

    coeffs = lhs.I*rhs
    return coeffs
