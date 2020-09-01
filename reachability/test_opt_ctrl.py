import numpy as np
import cv2
from copy import deepcopy
from scipy.io import loadmat


grid_max_X = 100
grid_max_Y = 100

grid = np.zeros([grid_max_Y, grid_max_X])
grid = np.stack([grid, grid, grid], axis=-1)


eps = 1e-5
dt = 0.002


fst = loadmat('RMAI_FO_dt01_t50_veryHigh_quadratic.mat')
eb = 1.2413
vf_X = fst['data_X']
vf_Y = fst['data_Y']
vf_dX = fst['derivX']
vf_dY = fst['derivY']


def random_goal():
    goal = [np.random.randint(20, 80), np.random.randint(20, 80)]
    return goal


def planner_model(p, goal):
    vec = [goal[0] - p[0], goal[1] - p[1]]
    if vec[0] > 0:
        bx = 2.0
    elif vec[0] < 0:
        bx = -2.0
    else:
        bx = 0.0
    if vec[1] > 0:
        by = 3.0
    elif vec[1] < 0:
        by = -3.0
    else:
        by = 0.0
    p_next = [p[0] + bx * dt, p[1] + by * dt]
    return p_next


def relative_model(s, p):
    r = deepcopy(s)
    r[0] -= p[0]
    r[2] -= p[1]
    return r


def pid_ctrl(s, goal):
    dx = goal[0] - s[0]
    dy = goal[1] - s[1]
    ax = 1.0 * dx
    ay = 1.0 * dy
    return [ax, ay, 0, 0]


def tracker_model(s, u):
    new_s = deepcopy(s)
    new_s[0] += s[1] - u[2] * dt
    new_s[1] += u[0] * dt
    new_s[2] += s[3] - u[3] * dt
    new_s[3] += u[1] * dt
    return new_s


def opt_ctrl(s, rnext):
    deriv1X = vf_dX[0][0][int(rnext[0]), int(rnext[1])]
    deriv2X = vf_dX[1][0][int(rnext[0]), int(rnext[1])]
    deriv1Y = vf_dY[0][0][int(rnext[2]), int(rnext[3])]
    deriv2Y = vf_dY[1][0][int(rnext[2]), int(rnext[3])]
    if deriv1X > 0:
        bx = -2.0
    else:
        bx = 2.0
    if deriv2X > 0:
        ax = 10.0
    else:
        ax = -10.0
    if deriv1Y > 0:
        by = -3.0
    else:
        by = 3.0
    if deriv2Y > 0:
        ay = 10.0
    else:
        ay = -10.0
    g = [s[1] - bx, ax, s[3] - by, ay]
    g[0] *= deriv1X
    g[1] *= deriv2X
    g[2] *= deriv1Y
    g[3] *= deriv2Y
    return g


p = [0, 0]
s = [0, 0, 0, 0]


while True:
    goal = random_goal()
    while abs(p[0] - goal[0]) >= eps and abs(p[1] - goal[1]) >= eps:
        pnext = planner_model(p, goal)
        rnext = relative_model(s, pnext)
        if max(vf_X[int(rnext[0]), int(rnext[1])], vf_Y[int(rnext[2]), int(rnext[3])]) >= eb:
            print("hehhe")
            u = opt_ctrl(s, rnext)
        else:
            u = pid_ctrl(s, goal)
        # visualize
        px = int(pnext[0])
        py = int(pnext[1])
        vgrid = deepcopy(grid)
        #vgrid[py, px, :] = [255, 0, 0]
        cv2.circle(vgrid, (px, py), 1, (255, 0, 0), thickness=1, lineType=8, shift=0) 
        sx = int(s[0])
        sy = int(s[2])
        vgrid[sy, sx, :] = [0, 0, 255]
        vgrid = cv2.resize(vgrid, (416, 416)).astype(np.uint8)
        cv2.imshow("Grid", vgrid)
        cv2.waitKey(50)
        # New states
        s = tracker_model(s, u)
        p = [s[0], s[2]]
