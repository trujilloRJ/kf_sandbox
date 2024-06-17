import numpy as np
from scipy.stats import multivariate_normal
from measurement import BaseMeas, MeasMixDoppler
from utils import *

def compute_Qdwpca(var_ax, var_ay, T):
    q = np.diag([var_ax, var_ay])
    gamma = np.array([
        [0.5*T**2, 0],
        [0,        0.5*T**2],
        [T,        0],
        [0,        T],
        [1,        0],
        [0,        1],
    ])
    return gamma@q@gamma.T


def compute_Qdwnca(var_vx, var_vy, T):
    q = np.diag([var_vx, var_vy])
    gamma = np.array([
        [0.5*T**2, 0],
        [0,        0.5*T**2],
        [T,        0],
        [0,        T],
    ])
    return gamma@q@gamma.T
    

class BaseKF:
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, compute_lk = True):
        try:
            z = z[:self.meas.dim_meas]
            R = self.meas.R
            est_meas, H_jac = self.meas.meas_fn(self.x)
        except:
            raise ValueError("Invalid measurement function")
        S = H_jac @ self.P @ H_jac.T + R
        K = self.P @ H_jac.T @ np.linalg.inv(S)
        y = (z - est_meas)
        self.x = self.x + K @ y
        self.P = self.P - K @ H_jac @ self.P
        if compute_lk:
            self.compute_likelihood(y, S)

    def compute_likelihood(self, y, S):
        self.likelihood = multivariate_normal.pdf(y, mean=np.zeros(len(y)), cov=S)


class LKF_CA(BaseKF):
    def __init__(self, cycle_time, meas_fn: BaseMeas, qvar_ax=1., qvar_ay=1.):
        T = cycle_time
        pvar_pos, pvar_vel, pvar_acc = 10**2, 5**2, 3**2
        self.dim_state = 6
        self.cyle_time = T
        self.F = self.get_state_transition(T)
        self.x = np.zeros(self.dim_state)
        self.P = np.diag([pvar_pos, pvar_pos, pvar_vel, pvar_vel, pvar_acc, pvar_acc])
        self.Q = compute_Qdwpca(qvar_ax, qvar_ay, T)
        self.meas = meas_fn
        self.likelihood = 0.

    def initialize_filter(self, state_init):
        self.x = state_init
    
    @staticmethod
    def get_state_transition(T):
        return np.array([ 
            [1, 0, T, 0, 0.5*T**2, 0       ],
            [0, 1, 0, T, 0,        0.5*T**2],
            [0, 0, 1, 0, T,        0       ],
            [0, 0, 0, 1, 0,        T       ],
            [0, 0, 0, 0, 1,        0       ],
            [0, 0, 0, 0, 0,        1       ],
        ])


class LKF_CV(BaseKF):
    def __init__(self, cycle_time, meas_fn, qvar_vx=1., qvar_vy=1.):
        T = cycle_time
        pvar_pos, pvar_vel = 10**2, 5**2
        self.dim_meas = 4
        self.dim_state = 4
        self.cyle_time = T
        self.F = np.array([ 
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.x = np.zeros(self.dim_state)
        self.P = np.diag([pvar_pos, pvar_pos, pvar_vel, pvar_vel])
        self.Q = compute_Qdwnca(qvar_vx, qvar_vy, T)
        self.meas_fn = meas_fn
        self.likelihood = 0.

    def initialize_filter(self, meas_init):
        self.x[:self.dim_meas] = meas_init


class IMM_CV_CA():
    # TODO: review after new measurement function update
    def __init__(self, cycle_time, t_mat, qvar_vx=1., qvar_vy=1., qvar_ax=1., qvar_ay=1.):
        meas_fn = MeasMixDoppler()
        self.dim_meas = 4
        self.dim_state = 6
        self.filters = [LKF_CV(cycle_time, meas_fn, qvar_vx, qvar_vy), LKF_CA(cycle_time, meas_fn, qvar_ax, qvar_ay)]
        self.IFCV, self.IFCA = 0, 1
        self.IVX, self.IVY, self.IAX, self.IAY = 2, 3, 4, 5
        self.n_filters = len(self.filters)
        self.mode_probs = 1/self.n_filters * np.ones(self.n_filters).T # initially all equally probable
        self.trans_mat = t_mat # transition matrix
        self.x = np.zeros(self.dim_state)
        self.P = np.zeros((self.dim_state, self.dim_state))
        self.likelihood = 0.

    def initialize_filter(self, meas_init):
        for filt in self.filters:
            filt.initialize_filter(meas_init)

    def _compute_cnorm(self):
        return self.trans_mat.T @ self.mode_probs
    
    def _compute_mixing_probs(self):
        cnorm = self._compute_cnorm()
        mix_probs = np.zeros((self.n_filters, self.n_filters))
        for i in range(self.n_filters):
            for j in range(self.n_filters):
                mix_probs[i, j] = self.trans_mat[i, j] * self.mode_probs[i] / cnorm[j]
        return mix_probs
    
    def _update_mode_probs(self):
        cnorm = self._compute_cnorm()
        pnorm = 0
        for j, filt in enumerate(self.filters):
            pnorm += filt.likelihood * cnorm[j]
        for j, filt in enumerate(self.filters):
            self.mode_probs[j] = filt.likelihood * cnorm[j] / pnorm

    def _augment_state(self):
        xCV, xCA = self.filters[self.IFCV].x, self.filters[self.IFCA].x
        pCV, pCA = self.filters[self.IFCV].P, self.filters[self.IFCA].P
        xCV_ = np.concatenate((xCV, xCA[self.IAX:].T))
        pCV_ = np.zeros_like(pCA)
        pCV_[0:self.IAX, 0:self.IAX] = pCV
        for i in [self.IAX, self.IAY]:
            pCV_[i, i] = pCA[i, i]
        return xCV_, xCA, pCV_, pCA

    def predict(self):
        mix_probs = self._compute_mixing_probs()

        # augment state of CV to match CA
        xCV_, xCA, pCV_, pCA = self._augment_state()

        # mixing
        xCVi = mix_probs[self.IFCV, self.IFCV]*xCV_ + mix_probs[self.IFCA, self.IFCV]*xCA 
        xCAi = mix_probs[self.IFCV, self.IFCA]*xCV_ + mix_probs[self.IFCA, self.IFCA]*xCA
        pCVi = mix_probs[self.IFCV, self.IFCV]*(pCV_ + np.outer(xCV_ - xCVi, xCV_ - xCVi)) \
               + mix_probs[self.IFCA, self.IFCV]*(pCA + np.outer(xCA - xCVi, xCA - xCVi))
        pCAi = mix_probs[self.IFCV, self.IFCA]*(pCV_ + np.outer(xCV_ - xCAi, xCV_ - xCAi)) \
               + mix_probs[self.IFCA, self.IFCA]*(pCA + np.outer(xCA - xCAi, xCA - xCAi))
        self.filters[self.IFCV].x, self.filters[self.IFCA].x = xCVi[:self.IAX], xCAi
        self.filters[self.IFCV].P, self.filters[self.IFCA].P = pCVi[:self.IAX, :self.IAX], pCAi

        # predict on each filter
        for filt in self.filters:
            filt.predict()

    def update(self, z, R):
        # update on each filter
        for filt in self.filters:
            filt.update(z, R, compute_lk = True)
        
        # update mode probabilities
        self._update_mode_probs()

        # combining states
        xCV_, xCA, pCV_, pCA = self._augment_state()
        x = self.mode_probs[self.IFCV] * xCV_ \
            + self.mode_probs[self.IFCA] * xCA
        P = self.mode_probs[self.IFCV]*(pCV_ + np.outer(xCV_ - x, xCV_ - x)) \
            + self.mode_probs[self.IFCA]*(pCA + np.outer(xCA - x, xCA - x))
        self.x, self.P = x, P

        
