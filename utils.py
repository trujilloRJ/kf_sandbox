import numpy as np

IX, IY, IVX, IVY, IAX, IAY = 0, 1, 2, 3, 4, 5 # indexes in state CA
IPHI, IV, IW = 2, 3, 4                       # indexes in state CTRV
IMR, IMPHI, IMD, IMX, IMY = 0, 1, 2, 3, 4          # indexes in measuremnents

def wrap_angle2(angle_rad: float):
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def circular_mean(angles: np.ndarray, weights: np.ndarray = None):
    if weights is None:
        weights = np.ones_like(angles)
    return np.arctan2(np.dot(np.sin(angles), weights), np.dot(np.cos(angles), weights))


# To make filterpy UKF work
def fx_ctrv(x: np.ndarray, T: float):
    w = x[IW]
    if np.abs(w) > np.radians(0.5):
        phi_p = x[IPHI] + w*T
        rc = x[IV]/w
        x[IX] += rc * (np.sin(phi_p) - np.sin(x[IPHI])) 
        x[IY] += rc * (-np.cos(phi_p) + np.cos(x[IPHI]))
        x[IPHI] = phi_p 
    else:
        x[IX] += x[IV]*np.cos(x[IPHI]) * T
        x[IY] += x[IV]*np.sin(x[IPHI]) * T
    return x

def hx_pos_only(x):
    x, y = x[IX], x[IY]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([r, phi])

def get_process_noise_matrix(x:np.ndarray, var_v, var_w, T: float):
    q = np.diag([var_v, var_w])
    phi = x[IPHI]
    gamma = np.array([
        [0.5*np.cos(phi)*T**2, 0],
        [0.5*np.sin(phi)*T**2, 0],
        [0, 0.5*T**2],
        [T, 0],
        [0, T],
    ])
    Q = gamma @ q @ gamma.T
    return Q