import math
import time
import numpy as np
import cv2
from copy import deepcopy


class Environment():

    def __init__(self, MAX_timestep=10000, visualize=False):
        """
            0 = floor
            1 = obstacles
            2 = opponent
            3 = agents
            4 = tracker error bound
            5 = opponent min error bound
            6 = opponent max error bound
        """
        self.MAX_timestep = MAX_timestep
        self.visualize = visualize
        self.safety_bound = 100.0             # tracker error bound
        self.min_opponent_bound = 100       # min opponent error bound
        self.max_opponent_bound = 600       # opponent error bound
        self.agent_width = 450           
        self.agent_height = 600           
        self.dt = 0.002                     # seconds for each timestep
        self.action_range = np.array([[2.0, 3.0, 4.16], [2.0, 3.0, 4.16]])

    def _get_placement_grid(self):
        placement_grid = np.zeros((4480, 8080))
        eps = round(self.safety_bound)
        # B1
        pos = [3380, 500]
        placement_grid[pos[0]-(100+eps):pos[0]+(100+eps), pos[1]-(500):pos[1]+(500+eps)] = 4
        placement_grid[pos[0]-(100):pos[0]+(100), pos[1]-(500):pos[1]+(500)] = 1
        # B2
        pos = [2240, 1900]
        placement_grid[pos[0]-(100+eps):pos[0]+(100+eps), pos[1]-(400+eps):pos[1]+(400+eps)] = 4
        placement_grid[pos[0]-(100):pos[0]+(100), pos[1]-(400):pos[1]+(400)] = 1
        # B3
        pos = [500, 1600]
        placement_grid[pos[0]-(500):pos[0]+(500+eps), pos[1]-(100+eps):pos[1]+(100+eps)] = 4
        placement_grid[pos[0]-(500):pos[0]+(500), pos[1]-(100):pos[1]+(100)] = 1
        # B4
        pos = [3445, 4040]
        placement_grid[pos[0]-(100+eps):pos[0]+(100+eps), pos[1]-(500+eps):pos[1]+(500+eps)] = 4
        placement_grid[pos[0]-(100):pos[0]+(100), pos[1]-(500):pos[1]+(500)] = 1
        # B5
        pos = [2240, 4040]
        placement_grid[pos[0]-(170+eps):pos[0]+(170+eps), pos[1]-(170+eps):pos[1]+(170+eps)] = 4
        placement_grid[pos[0]-(170):pos[0]+(170), pos[1]-(170):pos[1]+(170)] = 1
        # B6
        pos = [1035, 4040]
        placement_grid[pos[0]-(100+eps):pos[0]+(100+eps), pos[1]-(500+eps):pos[1]+(500+eps)] = 4
        placement_grid[pos[0]-(100):pos[0]+(100), pos[1]-(500):pos[1]+(500)] = 1
        # B7
        pos = [3980, 6480]
        placement_grid[pos[0]-(500+eps):pos[0]+(500+eps), pos[1]-(100+eps):pos[1]+(100+eps)] = 4
        placement_grid[pos[0]-(500):pos[0]+(500), pos[1]-(100):pos[1]+(100)] = 1
        # B8
        pos = [2240, 6180]
        placement_grid[pos[0]-(100+eps):pos[0]+(100+eps), pos[1]-(400+eps):pos[1]+(400+eps)] = 4
        placement_grid[pos[0]-(100):pos[0]+(100), pos[1]-(400):pos[1]+(400)] = 1
        # B9
        pos = [1100, 7580]
        placement_grid[pos[0]-(100+eps):pos[0]+(100+eps), pos[1]-(500+eps):pos[1]+(500+eps)] = 4
        placement_grid[pos[0]-(100):pos[0]+(100), pos[1]-(500):pos[1]+(500)] = 1
        return placement_grid

    def get_random_opponent_position_and_fill(self, grid, width, height, boundary):
        collided = True
        while collided:
            pos = [np.random.uniform() * 4480, np.random.uniform() * 8080, np.random.uniform() * 6.28319]
            pts_in_bound = self.points_in_rectangle(width, height, pos[1], pos[0], pos[2])
            collided = False
            for pts in pts_in_bound:
                x = pts[0]
                y = pts[1]
                if x < 0 or y < 0 or x >= 8080 or y >= 4480:
                    collided = True
                    break
                if grid[y, x] == 1 or grid[y, x] == 2:
                    collided = True
                    break
        pts_in_bound2 = self.points_in_rectangle(width+boundary, height+boundary, pos[1], pos[0], pos[2])
        cols, rows = zip(*pts_in_bound2)
        grid[rows, cols] = 5
        cols, rows = zip(*pts_in_bound)
        grid[rows, cols] = 2
        return pos, grid

    def get_random_position_and_fill(self, grid, width, height):
        collided = True
        while collided:
            pos = [np.random.uniform() * 4480, np.random.uniform() * 8080, np.random.uniform() * 6.28319]
            pts_in_bound = self.points_in_rectangle(width, height, pos[1], pos[0], pos[2])
            collided = False
            for pts in pts_in_bound:
                x = pts[0]
                y = pts[1]
                if x < 0 or y < 0 or x >= 8080 or y >= 4480:
                    collided = True
                    break
                if grid[y, x] != 0:
                    collided = True
                    break
        pts_in_bound2 = self.points_in_rectangle(width+self.safety_bound, height+self.safety_bound, pos[1], pos[0], pos[2])
        cols, rows = zip(*pts_in_bound2)
        grid[rows, cols] = 4
        cols, rows = zip(*pts_in_bound)
        grid[rows, cols] = 3
        return pos, grid, pts_in_bound, pts_in_bound2
      
    def get_random_position(self, grid):
        collided = True
        while collided:
            pos = [np.random.uniform() * 4480, np.random.uniform() * 8080, np.random.uniform() * 6.28319]
            pts_in_bound = self.points_in_rectangle(self.agent_width, self.agent_height, pos[1], pos[0], pos[2])
            collided = False
            for pts in pts_in_bound:
                x = pts[0]
                y = pts[1]
                if x < 0 or y < 0 or x >= 8080 or y >= 4480:
                    collided = True
                    break
                if x >= 0 and y >= 0 and grid[y, x] != 0:
                    collided = True
                    break
        return pos

    def points_in_circle(self, radius, x0=0, y0=0):
        x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
        y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
        x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
        pts = []
        for x, y in zip(x_[x], y_[y]):
            pts.append([x, y])
        pts = np.array(pts).astype(int)
        inidx = np.all(np.logical_and(np.array([0, 0]) <= pts, pts < np.array([8080, 4480])), axis=1)
        pts = pts[inidx]
        return pts

    def points_in_rectangle(self, width, height, x0=0, y0=0, theta0=0):
        x = np.arange(-width//2 - 1, width//2 + 1, dtype=int)
        y = np.arange(-height//2 - 1, height//2 + 1, dtype=int)
        px, py = np.meshgrid(x, y)
        px = px.flatten()
        py = py.flatten()
        qx = np.cos(theta0) * px - np.sin(theta0) * py
        qy = np.sin(theta0) * px + np.cos(theta0) * py
        qx = qx + x0
        qy = qy + y0
        pts = np.stack([qx, qy], axis=-1).astype(int)
        pts = np.unique(pts, axis=0)
        inidx = np.all(np.logical_and(np.array([0, 0]) <= pts, pts < np.array([8080, 4480])), axis=1)
        pts = pts[inidx]
        return pts
    
    def check_collision(self, pos, grid, agent_idx):
        #pts_in_bound = self.points_in_rectangle(self.agent_width, self.agent_height, pos[1], pos[0], pos[2])
        pts_in_bound = self.agents_pts[agent_idx]
        inidx = np.all(np.logical_and(np.array([0, 0]) <= pts_in_bound, pts_in_bound < np.array([8080, 4480])), axis=1)
        pts_in_bound = pts_in_bound[inidx]
        pts_in_bound = np.round(pts_in_bound).astype(int)
        cols, rows = zip(*pts_in_bound)
        grid_bool = grid[rows, cols]
        grid_bool = grid_bool[grid_bool != 0]
        for pts in pts_in_bound:
            x = int(round(pts[0]))
            y = int(round(pts[1]))
            if grid[y, x] != 0:
                return True
        return False
    
    def superpose_agents_on_grid(self, grid):
        for agent_pos in self.agents_pos:
            pts = self.points_in_rectangle(self.agent_width, self.agent_height, agent_pos[1], agent_pos[0], agent_pos[2])
            pts2 = self.points_in_rectangle(self.agent_width + self.safety_bound, self.agent_height + self.safety_bound,
                                            agent_pos[1], agent_pos[0], agent_pos[2])
            cols, rows = zip(*pts2)
            grid[rows, cols] = 4
            cols, rows = zip(*pts)
            grid[cols, rows] = 3
        return grid

    def superpose_max_opponent_bound(self, grid):
        for opponent_pos in self.opponents_pos:
            pts = self.points_in_circle(self.max_opponent_bound, opponent_pos[1], opponent_pos[0])
            pts2 = self.points_in_circle(int(self.max_opponent_bound * 0.9), opponent_pos[1], opponent_pos[0])
            dims = np.maximum(pts2.max(0),pts.max(0))+1
            pts = pts[~np.in1d(np.ravel_multi_index(pts.T,dims),np.ravel_multi_index(pts2.T,dims))]
            cols, rows = zip(*pts)
            grid[rows, cols] = 6
        return grid

    def observation(self):
        obs = np.array([self.agents_pos[0][0], self.agents_pos[0][1], self.agents_pos[0][2],
                              self.agents_pos[1][0], self.agents_pos[1][1], self.agents_pos[1][2],
                              self.opponents_pos[0][0], self.opponents_pos[0][1],
                              self.opponents_pos[1][0], self.opponents_pos[1][1]])
        return obs

    def dynamics(self, pos, action, idx):
        dx = action[0] * np.cos(pos[2]) + action[1] * np.sin(pos[2])
        dy = action[1] * np.cos(pos[2]) - action[0] * np.sin(pos[2])
        dtheta = action[2]
        dx *= self.dt
        dy *= self.dt
        dtheta *= self.dt
        new_theta = pos[2] + dtheta
        if new_theta < 0:
            new_theta = 6.28319 - new_theta
        elif new_theta > 6.28319:
            new_theta = new_theta - 6.28319
        new_pos = [pos[0] + dy, pos[1] + dx, new_theta]
        self.agents_pts[idx][:, 0] += dx
        self.agents_pts[idx][:, 1] += dy
        self.agents_safety_pts[idx][:, 0] += dx
        self.agents_safety_pts[idx][:, 1] += dy
        return new_pos

    def reward(self):
        rew = [0.0 for _ in range(2)]
        grid = deepcopy(self.placement_grid)
        #grid = self.superpose_agents_on_grid(grid)
        for idx in range(2):
            # check for collision
            collided = self.check_collision(self.agents_pos[idx], grid, idx)
            if collided:
                return -100.0, False
            # check for distance with opponent agents
            for opponent_pos in self.opponents_pos:
                dist = np.linalg.norm(np.array(self.agents_pos[idx])[0:2]**2 - np.array(opponent_pos)[0:2]**2)
                # penalize for getting into max error-bound of opponent agents
                if dist <= self.max_opponent_bound:
                    rew[idx] += -(1.0 / dist)
            # reward for getting closer to goal
            dist_2_goal = np.linalg.norm(np.array(self.agents_pos[idx])[0:2]**2 - np.array(self.goals[idx])**2)
            rew[idx] += -dist_2_goal
        return rew, False

    def step(self, action):
        rew, done = self.reward()
        # clip action
        action = np.clip(action, -self.action_range, self.action_range)
        action[:, 0:2] *= 1000.0
        for idx in range(2):
            self.agents_pos[idx] = self.dynamics(self.agents_pos[idx], action[idx], idx)
        self.timestep += 1
        if self.timestep >= self.MAX_timestep:
            done = True
        obs = self.observation()
        return obs, rew, done, dict()

    def reset(self):
        # Get placement grid
        grid = self._get_placement_grid()
        # Get agents' initial position
        opponent1_pos, grid = self.get_random_opponent_position_and_fill(grid, self.agent_width, self.agent_height, self.min_opponent_bound)
        opponent2_pos, grid = self.get_random_opponent_position_and_fill(grid, self.agent_width, self.agent_height, self.min_opponent_bound)
        virtual_grid = deepcopy(grid)
        agent1_pos, virtual_grid, agent1_pts, agent1_safety_pts = self.get_random_position_and_fill(virtual_grid, self.agent_width, self.agent_height)
        agent2_pos, virtual_grid, agent2_pts, agent2_safety_pts = self.get_random_position_and_fill(virtual_grid, self.agent_width, self.agent_height)
        self.agents_pos = [agent1_pos, agent2_pos]
        self.opponents_pos = [opponent1_pos, opponent2_pos]
        self.agents_pts = [agent1_pts.astype(np.float32), agent2_pts.astype(np.float32)]
        self.agents_safety_pts = [agent1_safety_pts.astype(np.float32), agent2_safety_pts.astype(np.float32)]
        self.goals = [[np.random.uniform() * 4480, np.random.uniform() * 8080],
                      [np.random.uniform() * 4480, np.random.uniform() * 8080]]
        self.placement_grid = grid
        self.timestep = 0
        return self.observation()
    
    def render(self):
        grid = deepcopy(self.placement_grid)
        grid = self.superpose_agents_on_grid(grid)
        grid = self.superpose_max_opponent_bound(grid)
        grid = np.stack([grid, grid, grid], axis=-1)
        grid[np.where((grid == [0, 0, 0]).any(axis=2))] = [169, 169, 169]   # floor
        grid[np.where((grid == [1, 1, 1]).any(axis=2))] = [0, 0, 0]   # obstacles
        grid[np.where((grid == [2, 2, 2]).any(axis=2))] = [255, 0, 0]   # opponents
        grid[np.where((grid == [3, 3, 3]).any(axis=2))] = [0, 0, 255]   # agents
        grid[np.where((grid == [4, 4, 4]).any(axis=2))] = [0, 165, 255]   # obstacles' tracker bound
        grid[np.where((grid == [5, 5, 5]).any(axis=2))] = [0, 255, 0]   # opponents' min bound
        grid[np.where((grid == [6, 6, 6]).any(axis=2))] = [128, 0, 128]   # opponents' max bound
        grid = cv2.resize(grid, (808, 448), interpolation=cv2.INTER_AREA).astype(np.uint8)
        grid = cv2.flip(grid, 0)    # flip because for opencv x-axis is flipped
        cv2.imshow("Grid", grid)
        cv2.waitKey(5)


def make_env(timesteps=10000, visualize=False):
    env = Environment(timesteps, visualize=visualize)
    #obs = env.reset()
    #action = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    #obs, rew, done, _ = env.step(action)
    return env


#make_env()
