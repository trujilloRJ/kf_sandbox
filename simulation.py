import numpy as np
from utils import *
from motion_models import MotionModel

def simulate_motion_CTRV(cycle_time, n_frames, x_init, motion_model: MotionModel, turn_frames):
    T = cycle_time
    dim_state = len(x_init)
    sim_state = np.zeros([dim_state, n_frames])
    for i in range(n_frames):
        if i==0:
            sim_state[:, i] = x_init.T
            continue
        
        x_next = motion_model.apply_transition_fn(sim_state[:, i-1], T)

        # add process noise
        Q = motion_model.get_process_noise_matrix(x_next, T)
        process_noise_state = np.random.multivariate_normal(dim_state*[0], Q)
        x_next += process_noise_state

        # add turn if present
        x_next[IW] = 0
        for t_seg in turn_frames:
            if (i >= t_seg[0]) & (i < t_seg[1]):
                x_next[IW] = t_seg[2]
                break

        sim_state[:, i] = x_next

    sim_state[IPHI, :] = wrap_angle2(sim_state[IPHI, :])
    
    return sim_state


# simulate measurement from a radar-like sensor
def simulate_measurements_polar(motion_cart: np.ndarray, std_r: float, std_phi: float, std_doppler: float):
    # true measurements in polar
    n_frames = motion_cart.shape[1]
    rho, phi = cart2pol(motion_cart[IX, :], motion_cart[IY, :])
    doppler = motion_cart[IVX] * np.cos(phi) + motion_cart[IVY] * np.sin(phi)

    # adding noise in polar
    rho += std_r * np.random.randn(n_frames)
    phi += std_phi * np.random.randn(n_frames)
    doppler += std_doppler * np.random.randn(n_frames)
    xm, ym = pol2cart(rho, phi)

    return np.array([
        rho,
        phi,
        doppler,
        xm,
        ym
    ])


def simulate_measurements_cartesian(motion_cart: np.ndarray, std_x: float, std_y: float, std_doppler: float):
    # true measurements in polar
    n_frames = motion_cart.shape[1]
    _, phi = cart2pol(motion_cart[IX, :], motion_cart[IY, :])
    doppler = motion_cart[IVX] * np.cos(phi) + motion_cart[IVY] * np.sin(phi)

    meas_x =  motion_cart[IX, :] + std_x * np.random.randn(n_frames)
    meas_y =  motion_cart[IY, :] + std_y * np.random.randn(n_frames)
    doppler += std_doppler * np.random.randn(n_frames)

    return np.array([
        meas_x,
        meas_y,
        doppler,
        meas_x,
        meas_y
    ])


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)