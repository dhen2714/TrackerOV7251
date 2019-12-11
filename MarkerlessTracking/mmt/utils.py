"""
Utility functions used by StereoFeatureTracker.
"""
import numpy as np
from mmt.camera_view import CameraView


def load_stereo_views(stereo_calib_file='Stereo_calibration.npz'):
    """
    Reads a stereo calibration npz file and returns two CameraView objects which
    can be loaded into StereoFeatureTracker.
    """
    f = np.load(stereo_calib_file)
    K1 = f['K1']
    K2 = f['K2']
    dc1 = np.squeeze(f['dc1'])
    dc2 = np.squeeze(f['dc2'])
    R = f['R']
    t = f['T']
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K2, np.hstack((R, t.reshape(3, 1))))
    fc1 = np.array([K1[0, 0], K1[1, 1]])
    fc2 = np.array([K2[0, 0], K2[1, 1]])
    pp1 = np.array([K1[0, 2], K1[1, 2]])
    pp2 = np.array([K2[0, 2], K2[1, 2]])
    kk1 = np.array([dc1[0], dc1[1], dc1[4]])
    kk2 = np.array([dc2[0], dc2[1], dc2[4]])
    kp1 = np.array([dc1[2], dc1[3]])
    kp2 = np.array([dc2[2], dc2[3]])
    view1 = CameraView(P1, fc1, pp1, kk1, kp1)
    view2 = CameraView(P2, fc2, pp2, kk2, kp2)
    return view1, view2


def mdot(*args):
    """Matrix multiplication for more than 2 arrays."""
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret


def mat2vec(H):
    """
    Converts a 4x4 representation of pose to a 6 vector.
    Inputs:
        H - 4x4 matrix.
    Outputs:
        v - [yaw,pitch,roll,x,y,z] (yaw,pitch,roll are in degrees)
    """
    sy = -H[2, 0]
    cy = 1-(sy*sy)

    if cy > 0.00001:
        cy = np.sqrt(cy)
        cx = H[2, 2]/cy
        sx = H[2, 1]/cy
        cz = H[0, 0]/cy
        sz = H[1, 0]/cy
    else:
        cy = 0.0
        cx = H[1, 1]
        sx = -H[1, 2]
        cz = 1.0
        sz = 0.0

    r2deg = (180/np.pi)
    return np.array([np.arctan2(sx, cx)*r2deg, np.arctan2(sy, cy)*r2deg,
                     np.arctan2(sz, cz)*r2deg,
                     H[0, 3], H[1, 3], H[2, 3]])


def vec2mat(*args):
    """
    Converts a six vector represenation of motion to a 4x4 matrix.
    Assumes yaw, pitch, roll are in degrees.
    Inputs:
        *args - either 6 numbers (yaw,pitch,roll,x,y,z) or an array with
             6 elements.
     Outputs:
        t     - 4x4 matrix representation of six vector.
    """
    if len(args) == 6:
        yaw = args[0]
        pitch = args[1]
        roll = args[2]
        x = args[3]
        y = args[4]
        z = args[5]
    elif len(args) == 1:
        yaw = args[0][0]
        pitch = args[0][1]
        roll = args[0][2]
        x = args[0][3]
        y = args[0][4]
        z = args[0][5]

    ax = np.radians(yaw)
    ay = np.radians(pitch)
    az = np.radians(roll)
    t1 = np.array([[1, 0, 0, 0],
                   [0, np.cos(ax), -np.sin(ax), 0],
                   [0, np.sin(ax), np.cos(ax), 0],
                   [0, 0, 0, 1]])
    t2 = np.array([[np.cos(ay), 0, np.sin(ay), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ay), 0, np.cos(ay), 0],
                   [0, 0, 0, 1]])
    t3 = np.array([[np.cos(az), -np.sin(az), 0, 0],
                   [np.sin(az), np.cos(az), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    tr = np.array([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return mdot(tr, t3, t2, t1)


def detect_outliers(array):
    """
    Removes outliers based on modified Z-score. Translated from Andre's IDL
    code.
    Inputs:
        array     - 1D array
    Outputs:
        outliers  - Array of indicies corresponding to outliers
    """
    med = np.median(array)
    MAD = np.median(np.abs(array - med))

    if MAD == 0:
        meanAD = np.mean(np.abs(array - med))
        if meanAD == 0:
            return np.array([])
        zscore = np.abs(array - med)/(1.253314*meanAD)
    else:
        zscore = np.abs(array - med)/(1.4826*MAD)
    return np.where(zscore > 3.5)[0]
