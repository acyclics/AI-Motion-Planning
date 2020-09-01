import numpy as np
from copy import deepcopy
from scipy.io import loadmat

from tracker import Tracker, Planner, Relative


class RmTracker(Tracker):

    def __init__(self):
        self.size = 5
        self.dt = 0.01

    def dynamics(self, x, u):
        new_x = deepcopy(x)
        new_x[0] += x[1] * self.dt#(x[1]*np.cos(x[4]) - x[3]*np.sin(x[4])) * self.dt
        new_x[1] += u[0] * self.dt
        new_x[2] += x[3] * self.dt#(x[1]*np.sin(x[4]) + x[3]*np.cos(x[4])) * self.dt
        new_x[3] += u[1] * self.dt
        new_x[4] += u[2] * self.dt
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
        x = p[0] + 2.0 * 0.01
        y = p[1] + 2.0 * 0.01
        theta = p[2] + 4.16 * 0.01
        return [x, y, theta]


class RmRelative(Relative):

    def __init__(self):
        self.size = 5
        self.uMax = np.array([10.0, 10.0, 8.32, 2.0, 3.0, 4.16])
        self.uMin = -self.uMax
    
    def state(self, s, p):
        r = deepcopy(s)
        r[0] -= p[0]
        r[2] -= p[1]
        r[4] -= p[2]
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
        # x-subsystem
        ax_deriv = deriv[1]
        uOpt_0 = (ax_deriv>=0)*self.uMin[0] + (ax_deriv<0)*self.uMin[0]
        bx_x_deriv = -deriv[0]*np.cos(x[4])
        uOpt_3_x = (bx_x_deriv>=0)*self.uMax[3] + (bx_x_deriv<0)*self.uMin[3]
        by_x_deriv = deriv[0]*np.sin(x[4])
        uOpt_4_x = (by_x_deriv>=0)*self.uMax[4] + (by_x_deriv<0)*self.uMin[4]
        # y-subsystem
        ay_deriv = deriv[3]
        uOpt_1 = (ay_deriv>=0)*self.uMin[1] + (ay_deriv<0)*self.uMax[1]
        bx_y_deriv = -deriv[2]*np.sin(x[4])
        uOpt_3_y = (bx_y_deriv>=0)*self.uMax[3] + (bx_y_deriv<0)*self.uMin[3]
        by_y_deriv = -deriv[2]*np.cos(x[4])
        uOpt_4_y = (by_y_deriv>=0)*self.uMax[4] + (by_y_deriv<0)*self.uMin[4]
        # pessimistic system
        w_deriv = deriv[4]
        uOpt_2 = (w_deriv>=0)*self.uMin[2] + (w_deriv<0)*self.uMax[2]
        btheta_deriv = -deriv[4]
        uOpt_5 = (btheta_deriv>=0)*self.uMax[5] + (btheta_deriv<0)*self.uMin[5]
        # Calculate opt ctrl
        uopt = [
            deriv[0] * ( x[1] - uOpt_3_x*np.cos(x[4]) + uOpt_4_x*np.sin(x[4]) ),
            deriv[1] * uOpt_0,
            deriv[2] * ( x[3] - uOpt_3_y*np.sin(x[4]) - uOpt_4_y*np.cos(x[4]) ),
            deriv[3] * uOpt_1,
            deriv[4] * ( uOpt_2 - uOpt_5 )
        ]
        return uopt


class Reach():

    def __init__(self):
        matlabf = "./RMAI_g_dt01_t5_medium_quadratic.mat"
        fst = loadmat(matlabf)
        
        eps = 0.1
        self.eb = max(fst['TEB_X'], fst['TEB_Y']) + eps
        self.vf_X = np.array(fst['data_X'])
        self.vf_Y = np.array(fst['data_Y'])

        self.vf_dX = fst['derivX']
        self.vf_dY = fst['derivY']

    def to_grid_index(self, r):
        r = deepcopy(r)
        r[0] = int(((r[0] + 5) / 10.0) * 60)
        r[1] = int(((r[1] + 2) / 4.0) * 60)
        r[2] = int(((r[2] + 5) / 10.0) * 60)
        r[3] = int(((r[3] + 3) / 6.0) * 60)
        r[4] = int(((r[4] + np.pi) / (2*np.pi)) * 60)
        return r

    def get_vf_dW(self, r):
        x_w = self.vf_X[r[0], r[1], r[4]]
        y_w = self.vf_Y[r[2], r[3], r[4]]
        if x_w >= y_w:
            dw = self.vf_dX[2][0][r[0], r[1], r[4]]
        else:
            dw = self.vf_dY[2][0][r[2], r[3], r[4]]
        return dw

    def check_on_boundary(self, r):
        r_int = [int(i) for i in r]
        vf_X_eb = self.vf_X[r_int[0], r_int[1], r_int[4]]
        vf_Y_eb = self.vf_Y[r_int[2], r_int[3], r_int[4]]
        vf_eb = max(vf_X_eb, vf_Y_eb)
        if vf_eb >= self.eb:
            return True
        else:
            return False

    def control(self, s, pnext):
        dx = pnext[0] - s[0]
        dy = pnext[1] - s[1]
        dw = pnext[2] - s[2]
        ax = 5.0 * dx
        ay = 5.0 * dy
        return [ax, ay, dw]
    
    def get_derivs(self, r):
        r_int = [int(i) for i in r]
        deriv = [
            self.vf_dX[0][0][r_int[0], r_int[1], r_int[4]],
            self.vf_dX[1][0][r_int[0], r_int[1], r_int[4]],
            self.vf_dY[0][0][r_int[2], r_int[3], r_int[4]],
            self.vf_dY[1][0][r_int[2], r_int[3], r_int[4]],
            self.get_vf_dW(r_int)
        ]
        return deriv
