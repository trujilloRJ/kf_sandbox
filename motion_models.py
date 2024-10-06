import numpy as np
from utils import *

class MotionModel():
    # base class
    def apply_transition_fn(self):
        pass

    def get_transition_matrix_jac(self):
        pass

    def get_process_noise_matrix(self):
        pass


class MM_CA(MotionModel):
    # Constant Acceleration
    # state: [x, y, vx, vy, ax, ay]
    def apply_transition_fn(self, x: np.ndarray, T: float):
        x[IX] += x[IVX]*T + 0.5*x[IAX]*T**2
        x[IY] += x[IVY]*T + 0.5*x[IAY]*T**2
        x[IVX] += x[IAX]*T
        x[IVY] += x[IAY]*T
        return x

    def get_transition_matrix_jac(self, T: float):
        return np.array([ 
            [1, 0, T, 0, 0.5*T**2, 0       ],
            [0, 1, 0, T, 0,        0.5*T**2],
            [0, 0, 1, 0, T,        0       ],
            [0, 0, 0, 1, 0,        T       ],
            [0, 0, 0, 0, 1,        0       ],
            [0, 0, 0, 0, 0,        1       ],
        ])
    
    def get_process_noise_matrix(self, T: float, var_ax: float, var_ay: float):
        Q = np.array([
            [T**5/20, 0,       T**4/8, 0,      T**3/6, 0],
            [0,       T**5/20, 0,      T**4/8, 0,      T**3/6],
            [T**4/8, 0,      T**3/3, 0,      T**2/2, 0],
            [0,      T**4/8, 0,      T**3/3, 0,      T**2/2],
            [T**3/6, 0,      T**2/2, 0,      T, 0],
            [0,      T**3/6, 0,      T**2/2, 0, T],
        ])
        for i in [IX, IVX, IAX]:
            Q[i, :] = var_ax * Q[i, :]
        for i in [IY, IVY, IAY]:
            Q[i, :] = var_ay * Q[i, :]
        return Q


class MM_CTRV(MotionModel):
    # Constant Turn Rate and Velocity
    # state: [x, y, v, phi, omega]
    # reference: https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/motion-model-design.html
    def __init__(self, var_v: float, var_w: float):
        self.dim_state = 5
        self.var_v = var_v
        self.var_w = var_w
        self.q = np.diag([var_v, var_w])

    def apply_transition_fn(self, x: np.ndarray, T: float):
        w = x[IW]
        if self._is_turning(w):
            phi_p = x[IPHI] + w*T
            rc = x[IV]/w
            x[IX] += rc * (np.sin(phi_p) - np.sin(x[IPHI])) 
            x[IY] += rc * (-np.cos(phi_p) + np.cos(x[IPHI]))
            x[IPHI] = phi_p 
        else:
            x[IX] += x[IV]*np.cos(x[IPHI]) * T
            x[IY] += x[IV]*np.sin(x[IPHI]) * T
        return x

    def get_transition_matrix_jac(self, x: np.ndarray, T: float):
        Fjac = np.eye(5)
        v, w, phi = x[IV], x[IW], x[IPHI]
        s, c = np.sin(phi), np.cos(phi)
        if self._is_turning(w):
            phi_p = phi + w*T
            rc = v/w
            sp, cp = np.sin(phi_p), np.cos(phi_p)
            Fjac[IX][IPHI] = rc * (-c + cp)
            Fjac[IY][IPHI] = rc * (-s + sp)
            Fjac[IX][IV] = 1/w * (-s + sp)
            Fjac[IY][IV] = 1/w * (c - cp)
            Fjac[IX][IW] = rc*T*cp - rc/w*(-s + sp)
            Fjac[IY][IW] = rc*T*sp - rc/w*(c - cp)
            Fjac[IPHI][IW] = T
        else:
            Fjac[IX][IPHI] = -v*s*T
            Fjac[IX][IV] = c*T
            # Fjac[IX][IW] = -v*s*T**2
            Fjac[IY][IPHI] = v*c*T
            Fjac[IY][IV] = s*T
            # Fjac[IY][IW] = v*c*T**2

        return Fjac

    def _is_turning(self, w):
        return np.abs(w) > np.radians(0.5)
    
    def get_process_noise_matrix(self, x:np.ndarray, T: float):
        phi = x[IPHI]
        gamma = np.array([
            [0.5*np.cos(phi)*T**2, 0],
            [0.5*np.sin(phi)*T**2, 0],
            [0, 0.5*T**2],
            [T, 0],
            [0, T],
        ])
        Q = gamma @ self.q @ gamma.T
        return Q
    


# # TODO: move this to motion models
# def compute_Qcwpa(var_x, var_y, T):
#     Q = np.array([
#         [T**5/20, 0,       T**4/8, 0,      T**3/6, 0],
#         [0,       T**5/20, 0,      T**4/8, 0,      T**3/6],
#         [T**4/8, 0,      T**3/3, 0,      T**2/2, 0],
#         [0,      T**4/8, 0,      T**3/3, 0,      T**2/2],
#         [T**3/6, 0,      T**2/2, 0,      T, 0],
#         [0,      T**3/6, 0,      T**2/2, 0, T],
#     ])
#     for i in [IX, IVX, IAX]:
#         Q[i, :] = var_x * Q[i, :]
#     for i in [IY, IVY, IAY]:
#         Q[i, :] = var_y * Q[i, :]
#     return Q

# def compute_Qcwna(var_x, var_y, T):
#     t1, t2, t3 = T, T**2/2, T**3/3
#     Q = np.array([
#         [t3, 0,  t2, 0],
#         [0,  t3, 0,  t2],
#         [t2, 0,  t1, 0],
#         [0,  t2, 0,  t1],
#     ])
#     for i in [IX, IVX]:
#         Q[i, :] = var_x * Q[i, :]
#     for i in [IY, IVY]:
#         Q[i, :] = var_y * Q[i, :]
#     return Q 

# def compute_Qdwpca(var_ax, var_ay, T):
#     q = np.diag([var_ax, var_ay])
#     gamma = np.array([
#         [0.5*T**2, 0],
#         [0,        0.5*T**2],
#         [T,        0],
#         [0,        T],
#         [1,        0],
#         [0,        1],
#     ])
#     return gamma@q@gamma.T

# def compute_Qdwnca(var_vx, var_vy, T):
#     q = np.diag([var_vx, var_vy])
#     gamma = np.array([
#         [0.5*T**2, 0],
#         [0,        0.5*T**2],
#         [T,        0],
#         [0,        T],
#     ])
#     return gamma@q@gamma.T
