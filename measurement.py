import numpy as np
from utils import *


class BaseMeas:
    def meas_fn(self, state):
        # TBI in subclass
        pass


class MeasMixDoppler(BaseMeas):
    def __init__(self, std_r: float, std_phi: float, std_doppler: float):
        self.dim_meas = 3
        self.R = np.diag([std_r**2, std_phi**2, std_doppler**2])

    def meas_fn(self, state):
        dim_state = len(state)
        x, y, vx, vy = state[IX], state[IY], state[IVX], state[IVY]

        # estimating measurement (Hx)
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        doppler = (x*vx + y*vy)/r
        est_meas = np.array([r, phi, doppler])

        # building jacobian
        H_jac = np.zeros((self.dim_meas, dim_state))
        H_jac[IR, IX] = x/r
        H_jac[IR, IY] = y/r
        H_jac[IT, IX] = -y/r**2
        H_jac[IT, IY] = x/r**2
        H_jac[ID, IX] = (y**2*vx - x*y*vy)/r**3
        H_jac[ID, IY] = (x**2*vy - x*y*vx)/r**3
        H_jac[ID, IVX] = x/r
        H_jac[ID, IVY] = y/r

        return est_meas, H_jac
    

class MeasMixPositionOnly(BaseMeas):
    def __init__(self, std_r: float, std_phi: float):
        self.dim_meas = 2
        self.R = np.diag([std_r**2, std_phi**2])

    def meas_fn(self, state):
        dim_state = len(state)
        x, y, vx, vy = state[IX], state[IY], state[IVX], state[IVY]

        # estimating measurement (Hx)
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        est_meas = np.array([r, phi])

        # building jacobian
        H_jac = np.zeros((self.dim_meas, dim_state))
        H_jac[IR, IX] = x/r
        H_jac[IR, IY] = y/r
        H_jac[IT, IX] = -y/r**2
        H_jac[IT, IY] = x/r**2

        return est_meas, H_jac
    