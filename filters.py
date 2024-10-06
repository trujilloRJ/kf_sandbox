import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky, inv
from motion_models import MM_CTRV
from measurement import MeasMixPositionOnly
from utils import *   

class BaseEKF:
    def _predict(self, fx, Fjac: np.ndarray, Q: np.ndarray):
        self.x = fx(self.x, self.T)
        self.P = Fjac @ self.P @ Fjac.T + Q

    def _update(self, y: np.ndarray, Hjac: np.ndarray, R: np.ndarray, compute_lk = False, compute_nis = False):
        S = Hjac @ self.P @ Hjac.T + R         # innovation covariance
        Sinv = np.linalg.inv(S)
        K = self.P @ Hjac.T @ Sinv              # Kalman gain
        self.x = self.x + K @ y
        self.P = self.P - K @ Hjac @ self.P
        # D = np.eye(*self.P.shape) - K @ Hjac
        # self.P = D @ self.P @ D.T + K @ R @ K.T
        if compute_lk:
            self.compute_likelihood(y, S)
        if compute_nis:
            self.nis = y.T @ Sinv @ y
        if not np.all(np.linalg.eigvals(S) > 0):
            print('S matrix not positive definite')

    def compute_likelihood(self, y, S):
        self.likelihood = multivariate_normal.pdf(y, mean=np.zeros(len(y)), cov=S)


class EKF_CTRV(BaseEKF):
    def __init__(self, cycle_time, var_v, var_w, std_r, std_phi):
        self.T = cycle_time
        self.mm = MM_CTRV(var_v, var_w)
        self.meas_fn = MeasMixPositionOnly(std_r, std_phi)
        self.fx = self.mm.apply_transition_fn
        self.dim_state = self.mm.dim_state
        self.dim_meas = self.meas_fn.dim_meas

    def initializeFilter(self, z0: np.ndarray):
        self.x = np.zeros(self.dim_state)
        self.x[IX] = z0[IX]
        self.x[IY] = z0[IY]
        self.P = np.diag([3**2, 3**2, np.radians(90)**2, 20**2, np.radians(10)**2])

    def predict(self):
        Fjac = self.mm.get_transition_matrix_jac(self.x, self.T)
        Q = self.mm.get_process_noise_matrix(self.x, self.T)
        self._predict(self.fx, Fjac, Q)
        self.x[IPHI] = wrap_angle2(self.x[IPHI])

    def update(self, z: np.ndarray):
        z_est, Hjac = self.meas_fn.meas_fn(self.x)
        y = z - z_est
        y[IMPHI] = wrap_angle2(y[IMPHI])
        R = self.meas_fn.compute_R_matrix()
        self._update(y, Hjac, R)
        self.x[IPHI] = wrap_angle2(self.x[IPHI])


class UKF_CTRV:
    def __init__(self, cycle_time, var_v, var_w, std_r, std_phi):
        self.T = cycle_time
        self.mm = MM_CTRV(var_v, var_w)
        self.meas_fn = MeasMixPositionOnly(std_r, std_phi)
        self.dim_state = self.mm.dim_state
        self.dim_meas = self.meas_fn.dim_meas
        self.n_sig = 2*self.dim_state + 1
        self.sigma = np.zeros((self.dim_state, self.n_sig)) 
        self._initialize_weights()
        self.x = np.zeros(self.dim_state)
        self.P = np.zeros((self.dim_state, self.dim_state))

    def initializeFilter(self, z0: np.ndarray):
        self.x[IX] = z0[IX]
        self.x[IY] = z0[IY]
        self.P = np.diag([3**2, 3**2, np.radians(90)**2, 20**2, np.radians(10)**2])

    def _initialize_weights(self):
        nx = self.dim_state
        kappa = 3 - nx
        alpha = 0.1
        beta = 2
        self.lambda_ = alpha**2 * (nx+kappa) - nx
        # self.lambda_ = 3 - L
        gamma = self.lambda_ + nx
        self.wm = np.full(self.n_sig, 1/(2*gamma))
        self.wc = np.full(self.n_sig, 1/(2*gamma))
        self.wm[0] = self.lambda_/gamma
        # self.wc[0] = self.lambda_/gamma
        self.wc[0] = self.lambda_/gamma + 1 - alpha**2 + beta

    def predict(self):
        nx, ns = self.dim_state, self.n_sig

        # augment state and covariance
        X_aug = self.x
        P_aug = self.P

        # generate sigma points
        L = cholesky(P_aug)
        S = np.zeros((nx, ns))
        S[:, 0] = X_aug
        for i in np.arange(1, nx + 1):
            S[:, i] = X_aug + np.sqrt(self.lambda_ + nx) * L[:, i-1]
            S[:, i+nx] = X_aug - np.sqrt(self.lambda_ + nx) * L[:, i-1]

        # predict sigma points
        S_pred = S
        for j in range(ns):
            S_pred[:, j] = self.mm.apply_transition_fn(S[:, j], self.T)

        # estimate mean and covariance of prediction
        x_mean, P_pred = self._estimate_mean_and_covariance(S_pred)
        Q = self.mm.get_process_noise_matrix(self.x, self.T)
        P_pred += Q

        self.sigma = S_pred
        self.x = x_mean
        self.P = P_pred

    def update(self, z):
        nz, nx, ns = self.dim_meas, self.dim_state, self.n_sig
        
        # predicting measurement
        Z_pred = np.zeros((nz, ns))
        for j in range(ns):
            Z_pred[:, j], _ = self.meas_fn.meas_fn(self.sigma[:, j])
        z_mean, Pzz = self._estimate_mean_and_covariance(Z_pred)
        R = self.meas_fn.compute_R_matrix()
        Pzz += R

        # computing Pxz
        Pxz = np.zeros((nx, nz))
        for j in range(ns):
            x_diff = self.sigma[:, j] - self.x
            x_diff[IPHI] = wrap_angle2(x_diff[IPHI])
            z_diff = Z_pred[:, j] - z_mean
            z_diff[IMPHI] = wrap_angle2(z_diff[IMPHI])
            Pxz += self.wc[j]*np.outer(x_diff, z_diff)

        # computing Kalman gain
        K = Pxz @ inv(Pzz)

        self.x += K @ (z - z_mean)
        self.P = self.P - K @ Pzz @ K.T
        
    def _estimate_mean_and_covariance(self, X_sig: np.ndarray):
        nrow, ncol = X_sig.shape
        x_mean = X_sig @ self.wm
        P_pred = np.zeros((nrow, nrow))
        for j in range(ncol):
            x_diff = X_sig[:, j] - x_mean
            P_pred += self.wc[j]*np.outer(x_diff, x_diff)
        return x_mean, P_pred



# class UKF_CTRV:
#     def __init__(self, cycle_time, var_v, var_w, std_r, std_phi):
#         self.T = cycle_time
#         self.mm = MM_CTRV(var_v, var_w)
#         self.meas_fn = MeasMixPositionOnly(std_r, std_phi)
#         self.dim_state = self.mm.dim_state
#         self.dim_meas = self.meas_fn.dim_meas
#         self.dim_aug = self.dim_state + 2
#         self.n_sig = 2*self.dim_aug + 1
#         self.sigma = np.zeros((self.dim_state, self.n_sig)) 
#         self._initialize_weights()
#         self.x = np.zeros(self.dim_state)
#         self.P = np.zeros((self.dim_state, self.dim_state))

#     def initializeFilter(self, z0: np.ndarray):
#         self.x[IX] = z0[IX]
#         self.x[IY] = z0[IY]
#         self.P = np.diag([3**2, 3**2, np.radians(90)**2, 20**2, np.radians(10)**2])

#     def _initialize_weights(self):
#         L = self.dim_aug
#         # kappa = 3 - L
#         # alpha = 0.1
#         # beta = 2
#         # self.lambda_ = alpha**2 * (L+kappa) - L
#         self.lambda_ = 3 - L
#         gamma = self.lambda_ + L
#         self.wm = np.full(self.n_sig, 1/(2*gamma))
#         self.wc = self.wm
#         self.wm[0] = self.lambda_/gamma
#         self.wc[0] = self.lambda_/gamma
#         # self.wc[0] = self.lambda_/gamma + 1 - alpha**2 + beta

#     def predict(self):
#         na, nx, ns = self.dim_aug, self.dim_state, self.n_sig

#         # augment state and covariance
#         X_aug = np.zeros(na)
#         X_aug[:nx] = self.x
#         P_aug = np.zeros((na, na))
#         P_aug[:nx, :nx] = self.P
#         P_aug[nx:, nx:] = self.mm.q

#         # generate sigma points
#         L = cholesky(P_aug)
#         S = np.zeros((na, ns))
#         S[:, 0] = X_aug
#         for i in np.arange(1, na + 1):
#             S[:, i] = X_aug + np.sqrt(self.lambda_ + na) * L[:, i-1]
#             S[:, i+na] = X_aug - np.sqrt(self.lambda_ + na) * L[:, i-1]
#         S[IPHI, :] = wrap_angle2(S[IPHI, :])

#         # predict sigma points
#         S_pred = S
#         for j in range(ns):
#             S_pred[:, j] = self.mm.apply_transition_fn(S[:, j], self.T)
#         S_pred[IPHI, :] = wrap_angle2(S_pred[IPHI, :])

#         # estimate mean and covariance of prediction
#         x_mean, P_pred = self._estimate_mean_and_covariance(S_pred)
#         x_mean[IPHI] = wrap_angle2(x_mean[IPHI])

#         self.sigma = S_pred[:nx, :]
#         self.x = x_mean[:nx]
#         self.P = P_pred[:nx, :nx]

#     def update(self, z):
#         nz, nx, ns = self.dim_meas, self.dim_state, self.n_sig
        
#         # predicting measurement
#         Z_pred = np.zeros((nz, ns))
#         for j in range(ns):
#             Z_pred[:, j], _ = self.meas_fn.meas_fn(self.sigma[:, j])
#         Z_pred[IMPHI, :] = wrap_angle2(Z_pred[IMPHI, :])
#         z_mean, Pzz = self._estimate_mean_and_covariance(Z_pred)
#         z_mean[IMPHI] = wrap_angle2(z_mean[IMPHI])
#         R = self.meas_fn.compute_R_matrix()
#         Pzz = Pzz + R

#         # computing Pxz
#         Pxz = np.zeros((nx, nz))
#         for j in range(ns):
#             x_diff = self.sigma[:, j] - self.x
#             x_diff[IPHI] = wrap_angle2(x_diff[IPHI])
#             z_diff = Z_pred[:, j] - z_mean
#             z_diff[IMPHI] = wrap_angle2(z_diff[IMPHI])
#             Pxz += self.wc[j]*np.outer(x_diff, z_diff)

#         # computing Kalman gain
#         K = Pxz @ inv(Pzz)

#         self.x += K @ (z - z_mean)
#         self.P = self.P - K @ Pzz @ K.T
        
#     def _estimate_mean_and_covariance(self, X_sig: np.ndarray):
#         nrow, ncol = X_sig.shape
#         x_mean = X_sig @ self.wm
#         P_pred = np.zeros((nrow, nrow))
#         for j in range(ncol):
#             x_diff = X_sig[:, j] - x_mean
#             P_pred += self.wc[j]*np.outer(x_diff, x_diff)
#         return x_mean, P_pred

        


# class IMM_CV_CA():
#     def __init__(self, cycle_time, t_mat, qvar_vx=1., qvar_vy=1., qvar_ax=1., qvar_ay=1.):
#         meas_fn = MeasMixDoppler()
#         self.dim_meas = 4
#         self.dim_state = 6
#         self.filters = [LKF_CV(cycle_time, meas_fn, qvar_vx, qvar_vy), LKF_CA(cycle_time, meas_fn, qvar_ax, qvar_ay)]
#         self.IFCV, self.IFCA = 0, 1
#         self.IVX, self.IVY, self.IAX, self.IAY = 2, 3, 4, 5
#         self.n_filters = len(self.filters)
#         self.mode_probs = 1/self.n_filters * np.ones(self.n_filters).T # initially all equally probable
#         self.trans_mat = t_mat # transition matrix
#         self.x = np.zeros(self.dim_state)
#         self.P = np.zeros((self.dim_state, self.dim_state))
#         self.likelihood = 0.

#     def initialize_filter(self, meas_init):
#         for filt in self.filters:
#             filt.initialize_filter(meas_init)

#     def _compute_cnorm(self):
#         return self.trans_mat.T @ self.mode_probs
    
#     def _compute_mixing_probs(self):
#         cnorm = self._compute_cnorm()
#         mix_probs = np.zeros((self.n_filters, self.n_filters))
#         for i in range(self.n_filters):
#             for j in range(self.n_filters):
#                 mix_probs[i, j] = self.trans_mat[i, j] * self.mode_probs[i] / cnorm[j]
#         return mix_probs
    
#     def _update_mode_probs(self):
#         cnorm = self._compute_cnorm()
#         pnorm = 0
#         for j, filt in enumerate(self.filters):
#             pnorm += filt.likelihood * cnorm[j]
#         for j, filt in enumerate(self.filters):
#             self.mode_probs[j] = filt.likelihood * cnorm[j] / pnorm

#     def _augment_state(self):
#         xCV, xCA = self.filters[self.IFCV].x, self.filters[self.IFCA].x
#         pCV, pCA = self.filters[self.IFCV].P, self.filters[self.IFCA].P
#         xCV_ = np.concatenate((xCV, xCA[self.IAX:].T))
#         pCV_ = np.zeros_like(pCA)
#         pCV_[0:self.IAX, 0:self.IAX] = pCV
#         for i in [self.IAX, self.IAY]:
#             pCV_[i, i] = pCA[i, i]
#         return xCV_, xCA, pCV_, pCA

#     def predict(self):
#         mix_probs = self._compute_mixing_probs()

#         # augment state of CV to match CA
#         xCV_, xCA, pCV_, pCA = self._augment_state()

#         # mixing
#         xCVi = mix_probs[self.IFCV, self.IFCV]*xCV_ + mix_probs[self.IFCA, self.IFCV]*xCA 
#         xCAi = mix_probs[self.IFCV, self.IFCA]*xCV_ + mix_probs[self.IFCA, self.IFCA]*xCA
#         pCVi = mix_probs[self.IFCV, self.IFCV]*(pCV_ + np.outer(xCV_ - xCVi, xCV_ - xCVi)) \
#                + mix_probs[self.IFCA, self.IFCV]*(pCA + np.outer(xCA - xCVi, xCA - xCVi))
#         pCAi = mix_probs[self.IFCV, self.IFCA]*(pCV_ + np.outer(xCV_ - xCAi, xCV_ - xCAi)) \
#                + mix_probs[self.IFCA, self.IFCA]*(pCA + np.outer(xCA - xCAi, xCA - xCAi))
#         self.filters[self.IFCV].x, self.filters[self.IFCA].x = xCVi[:self.IAX], xCAi
#         self.filters[self.IFCV].P, self.filters[self.IFCA].P = pCVi[:self.IAX, :self.IAX], pCAi

#         # predict on each filter
#         for filt in self.filters:
#             filt.predict()

#     def update(self, z, R):
#         # update on each filter
#         for filt in self.filters:
#             filt.update(z, R, compute_lk = True)
        
#         # update mode probabilities
#         self._update_mode_probs()

#         # combining states
#         xCV_, xCA, pCV_, pCA = self._augment_state()
#         x = self.mode_probs[self.IFCV] * xCV_ \
#             + self.mode_probs[self.IFCA] * xCA
#         P = self.mode_probs[self.IFCV]*(pCV_ + np.outer(xCV_ - x, xCV_ - x)) \
#             + self.mode_probs[self.IFCA]*(pCA + np.outer(xCA - x, xCA - x))
#         self.x, self.P = x, P