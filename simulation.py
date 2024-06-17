import numpy as np
from utils import *

# simulating scenario, use LKF_CA F matrix for transition
def simulate_motion(state_transition: np.ndarray, dim_state, n_frames, x_init, ax_periods=None, ay_periods=None):
    if ax_periods is None:
        ax_periods = [0, 0, 0]
    if ay_periods is None:
        ay_periods = [0, 0, 0]

    sim_state = np.zeros([dim_state, n_frames])
    for i in range(n_frames):
        if i==0:
            sim_state[:, i] = x_init.T
        else:
            sim_state[:, i] = state_transition @ sim_state[:, i-1]
        sim_state[IAX, i] = ax_periods[2] if (i >= ax_periods[0]) & (i <= ax_periods[1]) else 0.
        sim_state[IAY, i] = ay_periods[2] if (i >= ay_periods[0]) & (i <= ay_periods[1]) else 0.
    return sim_state


# simulate measurement from a radar-like sensor
def simulate_measurements(motion_cart: np.ndarray, std_r: float, std_phi: float, std_doppler: float):
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


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)