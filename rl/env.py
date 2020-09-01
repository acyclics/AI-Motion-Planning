import math
import time
import numpy as np
import cv2
from copy import deepcopy

from rl.sat import separating_axis_theorem


def checkEqual(lst):
   return lst[1:] == lst[:-1]


def normalize_angles(angles):
    '''Puts angles in [-pi, pi] range.'''
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -(np.pi + 1e-6) <= angles.min() and angles.max() <= (np.pi + 1e-6)
    return angles


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
        self.safety_bound = 100.0             # tracker error bound
        self.min_opponent_bound = 100       # min opponent error bound
        self.max_opponent_bound = 600       # opponent error bound
        self.agent_width = 450           
        self.agent_height = 600           
        self.dt = 0.01                     # seconds for each timestep
        self.action_range = np.array([[2.0, 3.0, 4.16], [2.0, 3.0, 4.16]])
        self.action_clip_range = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.visualize_obstacles = self._get_obstacles_noBound()
        self.visualize = visualize

    def _get_obstacles(self):
        eps = round(self.safety_bound)
        b1 = [500, 3380]
        b2 = [1900, 2240]
        b3 = [1600, 500]
        b4 = [4040, 3445]
        b5 = [4040, 2240]
        b6 = [4040, 1035]
        b7 = [6480, 3980]
        b8 = [6180, 2240]
        b9 = [7580, 1100]
        obstacles = [
            # ul, ur, br, bl
            [[b1[0]-500, b1[1]+(100+eps)], [b1[0]+(500+eps), b1[1]+(100+eps)], [b1[0]+(500+eps), b1[1]-(100+eps)], [b1[0]-500, b1[1]-(100+eps)]],
            [[b2[0]-(400+eps), b2[1]+(100+eps)], [b2[0]+(400+eps), b2[1]+(100+eps)], [b2[0]+(400+eps), b2[1]-(100+eps)], [b2[0]-(400+eps), b2[1]-(100+eps)]],
            [[b3[0]-(100+eps), b3[1]+(500+eps)], [b3[0]+(100+eps), b3[1]+(500+eps)], [b3[0]+(100+eps), b3[1]-(500)], [b3[0]-(100+eps), b3[1]-(500)]],
            [[b4[0]-(500+eps), b4[1]+(100+eps)], [b4[0]+(500+eps), b4[1]+(100+eps)], [b4[0]+(500+eps), b4[1]-(100+eps)], [b4[0]-(500+eps), b4[1]-(100+eps)]],
            [[b5[0]-(170+eps), b5[1]+(170+eps)], [b5[0]+(170+eps), b5[1]+(170+eps)], [b5[0]+(170+eps), b5[1]-(170+eps)], [b5[0]-(170+eps), b5[1]-(170+eps)]],
            [[b6[0]-(500+eps), b6[1]+(100+eps)], [b6[0]+(500+eps), b6[1]+(100+eps)], [b6[0]+(500+eps), b6[1]-(100+eps)], [b6[0]-(500+eps), b6[1]-(100+eps)]],
            [[b7[0]-(100+eps), b7[1]+(500)], [b7[0]+(100+eps), b7[1]+(500)], [b7[0]+(100+eps), b7[1]-(500+eps)], [b7[0]-(100+eps), b7[1]-(500+eps)]],
            [[b8[0]-(400+eps), b8[1]+(100+eps)], [b8[0]+(400+eps), b8[1]+(100+eps)], [b8[0]+(400+eps), b8[1]-(100+eps)], [b8[0]-(400+eps), b8[1]-(100+eps)]],
            [[b9[0]-(500+eps), b9[1]+(100+eps)], [b9[0]+(500), b9[1]+(100+eps)], [b9[0]+(500), b9[1]-(100+eps)], [b9[0]-(500+eps), b9[1]-(100+eps)]]
        ]
        return obstacles
    
    def _get_obstacles_noBound(self):
        eps = 0
        b1 = [500, 3380]
        b2 = [1900, 2240]
        b3 = [1600, 500]
        b4 = [4040, 3445]
        b5 = [4040, 2240]
        b6 = [4040, 1035]
        b7 = [6480, 3980]
        b8 = [6180, 2240]
        b9 = [7580, 1100]
        obstacles = [
            # ul, ur, br, bl
            [[b1[0]-500, b1[1]+(100+eps)], [b1[0]+(500+eps), b1[1]+(100+eps)], [b1[0]+(500+eps), b1[1]-(100+eps)], [b1[0]-500, b1[1]-(100+eps)]],
            [[b2[0]-(400+eps), b2[1]+(100+eps)], [b2[0]+(400+eps), b2[1]+(100+eps)], [b2[0]+(400+eps), b2[1]-(100+eps)], [b2[0]-(400+eps), b2[1]-(100+eps)]],
            [[b3[0]-(100+eps), b3[1]+(500+eps)], [b3[0]+(100+eps), b3[1]+(500+eps)], [b3[0]+(100+eps), b3[1]-(500)], [b3[0]-(100+eps), b3[1]-(500)]],
            [[b4[0]-(500+eps), b4[1]+(100+eps)], [b4[0]+(500+eps), b4[1]+(100+eps)], [b4[0]+(500+eps), b4[1]-(100+eps)], [b4[0]-(500+eps), b4[1]-(100+eps)]],
            [[b5[0]-(170+eps), b5[1]+(170+eps)], [b5[0]+(170+eps), b5[1]+(170+eps)], [b5[0]+(170+eps), b5[1]-(170+eps)], [b5[0]-(170+eps), b5[1]-(170+eps)]],
            [[b6[0]-(500+eps), b6[1]+(100+eps)], [b6[0]+(500+eps), b6[1]+(100+eps)], [b6[0]+(500+eps), b6[1]-(100+eps)], [b6[0]-(500+eps), b6[1]-(100+eps)]],
            [[b7[0]-(100+eps), b7[1]+(500)], [b7[0]+(100+eps), b7[1]+(500)], [b7[0]+(100+eps), b7[1]-(500+eps)], [b7[0]-(100+eps), b7[1]-(500+eps)]],
            [[b8[0]-(400+eps), b8[1]+(100+eps)], [b8[0]+(400+eps), b8[1]+(100+eps)], [b8[0]+(400+eps), b8[1]-(100+eps)], [b8[0]-(400+eps), b8[1]-(100+eps)]],
            [[b9[0]-(500+eps), b9[1]+(100+eps)], [b9[0]+(500), b9[1]+(100+eps)], [b9[0]+(500), b9[1]-(100+eps)], [b9[0]-(500+eps), b9[1]-(100+eps)]]
        ]
        return obstacles

    def get_rectangle(self, pos, width, height):
        hwidth = width // 2
        hheight = height // 2
        px = np.array([-hwidth, hwidth, hwidth, -hwidth])
        py = np.array([hheight, hheight, -hheight, -hheight])
        qx = np.cos(pos[2]) * px - np.sin(pos[2]) * py
        qy = np.sin(pos[2]) * px + np.cos(pos[2]) * py
        qx += pos[0]
        qy += pos[1]
        rect = np.stack([qx, qy], axis=-1)
        return rect
      
    def get_random_position(self, obstacles, width, height, bound):
        collided = True
        while collided:
            pos = [np.random.uniform() * 8080, np.random.uniform() * 4480, np.random.uniform() * 6.28319]
            rect = self.get_rectangle(pos, width, height)
            out_of_bound = False
            for r in rect:
                x = r[0]
                y = r[1]
                if x < 0 or y < 0 or x >= 8080 or y >= 4480:
                    out_of_bound = True
                    break
            if out_of_bound:
                collided = True
                continue
            collided = False
            for obstacle in obstacles:
                separated = self.check_rectangle_intersect(rect, obstacle)
                if not separated:
                    collided = True
                    break
        rect = self.get_rectangle(pos, width+bound, height+bound)
        obstacles.append(rect)
        return pos, obstacles

    def check_rectangle_intersect(self, rect1, rect2):
        rect1 = np.array(rect1)
        rect2 = np.array(rect2)
        return not separating_axis_theorem(rect1, rect2)

    def check_collision(self, pos, obstacles):
        rect = self.get_rectangle(pos, self.agent_width, self.agent_height)
        for r in rect:
            x = r[0]
            y = r[1]
            if x < 0 or y < 0 or x >= 8080 or y >= 4480:
                return True
        for obstacle in obstacles:
            separated = self.check_rectangle_intersect(rect, obstacle)
            if not separated:
                return True
        return False
    
    def observation(self):
        agents_pos = np.array(self.agents_pos)
        agents_pos[:, 0] /= 8080
        agents_pos[:, 1] /= 4480
        agents_pos[:, 2] = normalize_angles(agents_pos[:, 2]) / np.pi

        opponents_pos = np.array(self.opponents_pos)
        opponents_pos[:, 0] /= 8080
        opponents_pos[:, 1] /= 4480
        opponents_pos[:, 2] = normalize_angles(opponents_pos[:, 2]) / np.pi

        goals = np.array(self.goals)
        goals[:, 0] /= 8080
        goals[:, 1] /= 4480

        obs = np.array([agents_pos[0][0], agents_pos[0][1], agents_pos[0][2],
                        agents_pos[1][0], agents_pos[1][1], agents_pos[1][2],
                        opponents_pos[0][0], opponents_pos[0][1], opponents_pos[0][2],
                        opponents_pos[1][0], opponents_pos[1][1], opponents_pos[1][2],
                        goals[0][0], goals[0][1], goals[1][0], goals[1][1]])
        return obs

    def dynamics(self, pos, action, idx):
        dx = action[0] * np.cos(pos[2]) - action[1] * np.sin(pos[2])
        dy = action[0] * np.sin(pos[2]) + action[1] * np.cos(pos[2])
        dtheta = action[2]
        dx *= self.dt
        dy *= self.dt
        dtheta *= self.dt
        new_theta = pos[2] + dtheta
        if new_theta < 0:
            new_theta = 6.28319 - new_theta
        elif new_theta > 6.28319:
            new_theta = new_theta - 6.28319
        new_pos = [pos[0] + dx, pos[1] + dy, new_theta]
        return new_pos

    def reward(self):
        rew = 0.0
        for idx in range(2):
            # check for collision between agents
            rect_idx = self.get_rectangle(self.agents_pos[idx], self.agent_width, self.agent_height)
            rect_jdx = self.get_rectangle(self.agents_pos[1 - idx], self.agent_width + self.safety_bound, self.agent_height + self.safety_bound)
            separated = self.check_rectangle_intersect(rect_idx, rect_jdx)
            if not separated:
                return -10.0, False
            # check for collision with obstacles including opponents
            collided = self.check_collision(self.agents_pos[idx], self.obstacles)
            if collided:
                return -10.0, False
            # check for distance with opponent agents
            for opponent_pos in self.opponents_pos:
                dist = np.linalg.norm(np.array(self.agents_pos[idx])[0:2] - np.array(opponent_pos)[0:2])
                # penalize for getting into max error-bound of opponent agents
                if dist <= self.max_opponent_bound:
                    rew += -(1.0 / dist)
                pass
            # reward for getting closer to goal
            dist_2_goal = np.linalg.norm(np.array(self.agents_pos[idx])[0:2] - np.array(self.goals[idx]))
            rew += -dist_2_goal / np.sqrt(8080**2 + 4480**2)
        return rew, False

    def step(self, action):
        rew, done = self.reward()
        # clip action
        action = np.clip(action, -self.action_clip_range, self.action_clip_range)
        action[:, 0] *= 2000.0
        action[:, 1] *= 3000.0
        action[:, 2] *= 4.16
        for idx in range(2):
            self.agents_pos[idx] = self.dynamics(self.agents_pos[idx], action[idx], idx)
        self.timestep += 1
        if self.timestep >= self.MAX_timestep:
            done = True
        obs = self.observation()
        return obs, rew, done, dict()

    def reset(self):
        obstacles = self._get_obstacles()
        # Get agents' initial position
        opponent1_pos, obstacles = self.get_random_position(obstacles, self.agent_width, self.agent_height, self.min_opponent_bound)
        opponent2_pos, obstacles = self.get_random_position(obstacles, self.agent_width, self.agent_height, self.min_opponent_bound)
        virtual_obstacles = deepcopy(obstacles)
        agent1_pos, virtual_obstacles = self.get_random_position(virtual_obstacles, self.agent_width, self.agent_height, self.safety_bound)
        agent2_pos, virtual_obstacles = self.get_random_position(virtual_obstacles, self.agent_width, self.agent_height, self.safety_bound)
        self.agents_pos = [agent1_pos, agent2_pos]
        self.opponents_pos = [opponent1_pos, opponent2_pos]
        self.goals = [[np.random.uniform() * 8080, np.random.uniform() * 4480],
                      [np.random.uniform() * 8080, np.random.uniform() * 4480]]
        self.obstacles = obstacles
        self.timestep = 0
        if self.visualize:
            self.image = self.get_background()
        return self.observation()
    
    def get_background(self):
        image = np.full((4480, 8080), 169)
        image = np.stack([image, image, image], axis=-1).astype(np.uint8)
        obstacles = deepcopy(self.obstacles)[:-2]
        # Draw tracker bound + obstacles first
        for obstacle in obstacles:
            rect = cv2.minAreaRect(np.array(obstacle).astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(image, [box], color=(0,165,255))
        # Then draw obstacles
        for obstacle in self.visualize_obstacles:
            rect = cv2.minAreaRect(np.array(obstacle).astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(image, [box], color=(0,0,0))
        # Display all opponents
        for opponent_pos in self.opponents_pos:
            # Draw min error bound first
            rect = self.get_rectangle(opponent_pos, self.agent_width + self.min_opponent_bound, self.agent_height + self.min_opponent_bound)
            rect = cv2.minAreaRect(np.array(rect).astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(image, [box], color=(0,255,0))
            # Then draw opponent
            rect = self.get_rectangle(opponent_pos, self.agent_width, self.agent_height)
            rect = cv2.minAreaRect(np.array(rect).astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(image, [box], color=(255,0,0))
            # Draw opponents' max error bound
            cv2.circle(image, (int(opponent_pos[0]), int(opponent_pos[1])), self.max_opponent_bound, (128, 0, 128), thickness=50)
        # Display goals
        cv2.circle(image, (int(self.goals[0][0]), int(self.goals[0][1])), 5, (255, 255, 0), thickness=100)
        cv2.circle(image, (int(self.goals[1][0]), int(self.goals[1][1])), 5, (0, 255, 255), thickness=100)
        return image

    def render(self):
        image = deepcopy(self.image)
        # Display all agents
        for agent_pos in self.agents_pos:
            # Draw tracker bound first
            rect = self.get_rectangle(agent_pos, self.agent_width + self.safety_bound, self.agent_height + self.safety_bound)
            rect = cv2.minAreaRect(np.array(rect).astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(image, [box], color=(0,165,255))
            # Then draw agents
            rect = self.get_rectangle(agent_pos, self.agent_width, self.agent_height)
            rect = cv2.minAreaRect(np.array(rect).astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(image, [box], color=(0,0,255))
        image = cv2.resize(image, (808, 448), interpolation=cv2.INTER_AREA)
        image = cv2.flip(image, 0)    # flip because for opencv x-axis is flipped
        cv2.imshow("image", image)
        cv2.waitKey(5)


def make_env(timesteps=10000, visualize=False):
    env = Environment(timesteps, visualize=visualize)
    #obs = env.reset()
    #action = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    #obs, rew, done, _ = env.step(action)
    return env


#make_env()
