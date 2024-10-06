import numpy as np

IX, IY, IVX, IVY, IAX, IAY = 0, 1, 2, 3, 4, 5 # indexes in state CA
IPHI, IV, IW = 2, 3, 4                       # indexes in state CTRV
IMR, IMPHI, IMD, IMX, IMY = 0, 1, 2, 3, 4          # indexes in measuremnents

def wrap_angle2(angle_rad: float):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
    # return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))