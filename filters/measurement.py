import numpy as np
from utils import *


class BaseMeas:
    def meas_fn(self, state):
        # TBI in subclass
        pass

    def compute_R_matrix(self, **kwargs):
        # To be re-implemented in child classes otherwise
        return self.R


class MeasConverted(BaseMeas):
    def __init__(self, std_r: float, std_phi: float):
        self.dim_meas = 2
        self.std_r = std_r
        self.std_phi = std_phi

    def meas_fn(self, state):
        dim_state = len(state)
        x, y = state[IX], state[IY]

        # estimating measurement (Hx)
        est_meas = np.array([x, y])

        # building jacobian (linear measurement function)
        H_jac = np.zeros((self.dim_meas, dim_state))
        H_jac[IX, IX] = 1
        H_jac[IY, IY] = 1

        return est_meas, H_jac
    
    def compute_R_matrix(self, **kwargs):
        r2, phi = kwargs.get('meas_r')**2, kwargs.get('meas_phi')
        s2_phi = self.std_phi**2
        s2_r = self.std_r**2
        cc, ss = np.cos(phi)**2, np.sin(phi)**2
        # linearize error
        var_x = r2 * s2_phi * ss + s2_r * cc
        var_y = r2 * s2_phi * cc + s2_r * ss
        var_xy = (s2_r - r2 * s2_phi) * np.sin(phi) * np.cos(phi)
        return np.array([
            [var_x, var_xy],
            [var_xy, var_y],
        ])
    

class MeasCartesianPos(BaseMeas):
    def __init__(self, std_x: float, std_y: float):
        self.dim_meas = 2
        self.R = np.diag([std_x**2, std_y**2])

    def meas_fn(self, state):
        dim_state = len(state)
        x, y = state[IX], state[IY]

        # estimating measurement (Hx)
        est_meas = np.array([x, y])

        # building jacobian (linear measurement function)
        H_jac = np.zeros((self.dim_meas, dim_state))
        H_jac[IX, IX] = 1
        H_jac[IY, IY] = 1

        return est_meas, H_jac
    
    def compute_R_matrix(self, **kwargs):
        return self.R



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
        H_jac[IMR, IX] = x/r
        H_jac[IMR, IY] = y/r
        H_jac[IMPHI, IX] = -y/r**2
        H_jac[IMPHI, IY] = x/r**2
        H_jac[IMD, IX] = vx/r - x/r**3 * (x*vx + y*vy)
        H_jac[IMD, IY] = vy/r - y/r**3 * (x*vx + y*vy)
        H_jac[IMD, IVX] = x/r
        H_jac[IMD, IVY] = y/r

        return est_meas, H_jac
    

class MeasMixPositionOnly(BaseMeas):
    def __init__(self, std_r: float, std_phi: float):
        self.dim_meas = 2
        self.R = np.diag([std_r**2, std_phi**2])

    def meas_fn(self, state):
        dim_state = len(state)
        x, y = state[IX], state[IY]

        # estimating measurement (Hx)
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        est_meas = np.array([r, phi])

        # building jacobian
        H_jac = np.zeros((self.dim_meas, dim_state))
        H_jac[IMR, IX] = x/r
        H_jac[IMR, IY] = y/r
        H_jac[IMPHI, IX] = -y/r**2
        H_jac[IMPHI, IY] = x/r**2

        return est_meas, H_jac
    
    def compute_R_matrix(self, **kwargs):
        return self.R
    