import numpy as np
from copy import deepcopy
from scipy.io import loadmat

from reachability.tracker import Tracker, Planner, Relative


def normalize_angles(angles):
    '''Puts angles in [-pi, pi] range.'''
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -(np.pi + 1e-6) <= angles.min() and angles.max() <= (np.pi + 1e-6)
    return angles


class RmTracker(Tracker):

    def __init__(self):
        self.size = 5
        self.dt = 0.01

    def dynamics(self, x, u):
        new_x = deepcopy(x)
        new_x[0] += (x[1]*np.cos(x[4]) - x[3]*np.sin(x[4])) * self.dt
        new_x[1] += u[0] * self.dt
        new_x[2] += (x[1]*np.sin(x[4]) + x[3]*np.cos(x[4])) * self.dt
        new_x[3] += u[1] * self.dt
        new_x[4] += u[2] * self.dt
        #new_x[4] = normalize_angles(np.array([new_x[4]]))[0]
        return new_x


class RmPlanner(Planner):

    def __init__(self):
        self.size = 3

    def dynamics(self, x, u):
        new_x = deepcopy(x)
        new_x[0] += u[0]*np.cos(x[2]) - u[1]*np.sin(x[2])
        new_x[1] += u[0]*np.sin(x[2]) + u[1]*np.cos(x[2])
        new_x[2] += u[2]
        return new_x
    
    def project(self, s):
        p = [s[0], s[2], s[4]]
        return p
    
    def control(self, p):
        x = 5
        y = 5
        theta = 0.7 #p[2] + 4.16 * 0.01
        #theta = normalize_angles(np.array([theta]))[0]
        return [x, y, theta]


class RmRelative(Relative):

    def __init__(self):
        self.size = 5
        self.uMax = np.array([10.0, 10.0, 6.0, 2.0, 3.0, 4.16])
        self.uMin = -self.uMax
    
    def clip_state(self, r):
        r[0] = np.clip(r[0], -5, 5)
        r[1] = np.clip(r[1], -2, 2)
        r[2] = np.clip(r[2], -5, 5)
        r[3] = np.clip(r[3], -3, 3)
        r[4] = np.clip(r[4], -np.pi, np.pi)
        return r

    def state(self, s, p):
        r = deepcopy(s)
        r[0] -= p[0]
        r[2] -= p[1]
        r[4] -= p[2]
        r = self.clip_state(r)
        return r

    def dynamics(self, r, u, d):
        rnext = deepcopy(r)
        rnext = np.clip(rnext, [-5, -2, -5, -3, -np.pi], [5, 2, 5, 3, np.pi])
        rnext[0] += x[1] - u[3]*np.cos(x[4]) + u[4]*np.sin(x[4]) + d[0]
        rnext[1] += u[0]
        rnext[2] += x[3] - u[3]*np.sin(x[4]) - u[4]*np.cos(x[4]) + d[1]
        rnext[3] += u[1]
        rnext[4] += u[2] - u[5] + d[2]
        return rnext
    
    def optControl(self, deriv, r, x):
        ax = (deriv[1]>=0)*self.uMin[0] + (deriv[1]<0)*self.uMax[0]
        ay = (deriv[3]>=0)*self.uMin[1] + (deriv[3]<0)*self.uMax[1]
        w = (deriv[4]>=0)*self.uMin[2] + (deriv[4]<0)*self.uMax[2]
        # Calculate opt ctrl
        uopt = [
            ax, ay, w
        ]
        return uopt


class Reach():

    def __init__(self):
        matlabf = "./reachability/RMAI_g_dt01_t5_medium_quadratic.mat"
        fst = loadmat(matlabf)
        
        eps = 0.1
        self.eb = np.max([fst['TEB_X1'], fst['TEB_X2'], fst['TEB_X3'], fst['TEB_X4']]) + eps

        self.vf_X1 = np.array(fst['data0_X1'])
        self.vf_X2 = np.array(fst['data0_X2'])
        self.vf_X3 = np.array(fst['data0_X3'])
        self.vf_X4 = np.array(fst['data0_X4'])

        self.vf_dX1 = fst['derivX1']
        self.vf_dX2 = fst['derivX2']
        self.vf_dX3 = fst['derivX3']
        self.vf_dX4 = fst['derivX4']
        
    def to_grid_index(self, r):
        r = deepcopy(r)
        r[0] = int(((r[0] + 5) / 10.0) * 60)
        r[1] = int(((r[1] + 2) / 4.0) * 60)
        r[2] = int(((r[2] + 5) / 10.0) * 60)
        r[3] = int(((r[3] + 3) / 6.0) * 60)
        r[4] = int(((r[4] + np.pi) / (2*np.pi)) * 60)
        return r

    def check_on_boundary(self, r):
        r_int = [int(i) for i in r]
        vf_X1_eb = self.vf_X1[r_int[0], r_int[1], r_int[4]]
        vf_X2_eb = self.vf_X2[r_int[0], r_int[3], r_int[4]]
        vf_X3_eb = self.vf_X3[r_int[1], r_int[2], r_int[4]]
        vf_X4_eb = self.vf_X4[r_int[2], r_int[3], r_int[4]]
        vf_eb = np.max([vf_X1_eb, vf_X2_eb, vf_X3_eb, vf_X4_eb])
        if vf_eb >= self.eb:
            return True
        else:
            return False

    def control(self, s, pnext):
        dx = pnext[0] - s[0]
        dy = pnext[1] - s[2]
        dw = pnext[2] - s[4]
        ax = 5.0 * dx
        ay = 5.0 * dy
        return [ax, ay, dw]
    
    def get_derivs(self, r):
        r_int = [int(i) for i in r]

        # All x deriv
        if self.vf_X1[r_int[0], r_int[1], r_int[4]] >= self.vf_X2[r_int[0], r_int[3], r_int[4]]:
            x_deriv = self.vf_dX1[0][0][r_int[0], r_int[1], r_int[4]]
        else:
            x_deriv = self.vf_dX2[0][0][r_int[0], r_int[3], r_int[4]]

        # All vx deriv
        if self.vf_X1[r_int[0], r_int[1], r_int[4]] >= self.vf_X3[r_int[1], r_int[2], r_int[4]]:
            vx_deriv = self.vf_dX1[1][0][r_int[0], r_int[1], r_int[4]]
        else:
            vx_deriv = self.vf_dX3[1][0][r_int[1], r_int[2], r_int[4]]

        # All y deriv
        if self.vf_X3[r_int[1], r_int[2], r_int[4]] >= self.vf_X4[r_int[2], r_int[3], r_int[4]]:
            y_deriv = self.vf_dX3[0][0][r_int[1], r_int[2], r_int[4]]
        else:
            y_deriv = self.vf_dX4[0][0][r_int[2], r_int[3], r_int[4]]

        # All vy deriv
        if self.vf_X2[r_int[0], r_int[3], r_int[4]] >= self.vf_X4[r_int[2], r_int[3], r_int[4]]:
            vy_deriv = self.vf_dX2[1][0][r_int[0], r_int[3], r_int[4]]
        else:
            vy_deriv = self.vf_dX4[1][0][r_int[2], r_int[3], r_int[4]]

        # All theta deriv
        theta_idx = np.argmax([
            self.vf_X1[r_int[0], r_int[1], r_int[4]],
            self.vf_X2[r_int[0], r_int[3], r_int[4]],
            self.vf_X3[r_int[1], r_int[2], r_int[4]],
            self.vf_X4[r_int[2], r_int[3], r_int[4]]
        ])
        theta_deriv = [
            self.vf_dX1[2][0][r_int[0], r_int[1], r_int[4]],
            self.vf_dX2[2][0][r_int[0], r_int[3], r_int[4]],
            self.vf_dX3[2][0][r_int[1], r_int[2], r_int[4]],
            self.vf_dX4[2][0][r_int[2], r_int[3], r_int[4]]
        ]
        theta_deriv = theta_deriv[theta_idx]

        deriv = [
            x_deriv,
            vx_deriv,
            y_deriv,
            vy_deriv,
            theta_deriv
        ]
        return deriv
