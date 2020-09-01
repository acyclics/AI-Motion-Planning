import numpy as np
import cv2
from copy import deepcopy


class FastTrack():

    def __init__(self, tracker, planner, relative, reach):
        self.tracker = tracker
        self.planner = planner
        self.relative = relative
        self.reach = reach
        
        grid_max_X = 50
        grid_max_Y = 50
        grid = np.zeros([grid_max_Y, grid_max_X])
        self.grid = np.stack([grid, grid, grid], axis=-1)
        
        self.eps = 1e-5
        self.dt = 0.01
    
    def random_goal(self):
        goal = [np.random.randint(20, 80), np.random.randint(20, 80)]
        return goal

    def run(self, its):
        p = [0 for _ in range(self.planner.size)]
        s = [0 for _ in range(self.tracker.size)]

        while True:
            #goal = self.random_goal()

            #while abs(p[0] - goal[0]) >= self.eps and abs(p[1] - goal[1]) >= self.eps:
            pnext = self.planner.control(p)
            rnext = self.relative.state(s, pnext)

            rnext = self.reach.to_grid_index(rnext)

            if self.reach.check_on_boundary(rnext):
                deriv = self.reach.get_derivs(rnext)
                u = self.relative.optControl(deriv, rnext, s)
            else:
                u = self.reach.control(s, pnext)

            # visualize
            vgrid = deepcopy(self.grid)

            px = int(pnext[0])
            py = int(pnext[1])
            cv2.circle(vgrid, (px, py), 1, (255, 0, 0), thickness=2, lineType=8, shift=0)

            sx = int(s[0])
            sy = int(s[2])
            vgrid[sy, sx, :] = [0, 0, 255]

            vgrid = cv2.resize(vgrid, (416, 416)).astype(np.uint8)
            cv2.imshow("Grid", vgrid)
            cv2.waitKey(5)

            # New states
            s = self.tracker.dynamics(s, u)
            p = self.planner.project(s)
